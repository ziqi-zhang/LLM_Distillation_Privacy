import time
import os

import torch
import torch.distributed as dist
import deepspeed
import copy
import json
from pdb import set_trace as st
from transformers import mpu, AutoModelForCausalLM, AutoTokenizer

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model
from data_utils.prompt_datasets import PromptDataset
from evaluate_mi_main import *
# from evaluate_exposure_bias import evaluate_eb, prepare_dataset_eb
from mi_utils import compute_AUC, plot_ROC, compute_tpr_25_05_01_001_0

torch.set_num_threads(4)

def wrap_model(args, model, ds_config):
    optimizer, lr_scheduler = None, None
        
    if args.model_type=="qwen" and ds_config['fp16']['enabled']==True:
        import copy
        ds_config['bf16']=copy.deepcopy(ds_config['fp16'])
        ds_config['fp16']['enabled']=False
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    return model

def setup_model(args, ds_config, device):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler

    model = wrap_model(args, model, ds_config)
    
    # get the memory usage
    # print_rank("Model mem\n", torch.cuda.memory_summary())
    return model

def prepare_dataset_mi(args, tokenizer):
    data = {}
    # Must first use full data -> select public data -> select dev number of data
    data["train"] = PromptDataset(args, tokenizer, "train", args.data_dir, -1)
    if args.train_data_mode is not "all":
        half_len = int(len(data["train"]) / 2)
        indixes = list(range(half_len)) if args.train_data_mode == "private" else list(range(half_len, len(data["train"])))
        data["train"].set_subset(indixes)
        log_str = f"Sample training dataset, train num {len(data['train'])}, mode {args.train_data_mode}, first ten indices {indixes[:10]}"
        save_rank(log_str, os.path.join(args.save, "train_log.txt"))
    # if args.dev_num % args.batch_size != 0:
    #     args.dev_num = (args.dev_num // args.batch_size + 1) * args.batch_size
    #     log_str = f"Dev num is not divisible by batch size, set dev num to {args.dev_num}"
    #     save_rank(log_str, os.path.join(args.save, "train_log.txt"))
    data["train"].set_num(args.dev_num)
    log_str = f"Sample training dataset, train num {len(data['train'])}, first ten real indices {[data['train'].data_idxs[i] for i in range(10)]}"
    save_rank(log_str, os.path.join(args.save, "train_log.txt"))
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)
    log_str = f"Sample test dataset, test num {len(data['test'])}, first ten real indices {[data['test'].data_idxs[i] for i in range(10)]}"

    return data


def run_model_memorization(args, tokenizer, model, dataset: PromptDataset, epoch, device, desc=None):
    
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    
    generation_config = GenerationConfig (
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=args.min_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    all_memorized_ids, all_valid_ids = [], []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=desc, disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)

            query_ids = model_batch["input_ids"]
            label_ids = no_model_batch["rest_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)

            if args.model_type == "opt":
                label_ids = label_ids[:, 1:]
            
            truncated_label_ids = label_ids[:, :32]
            truncated_response_ids = response_ids[:, :32]
            truncated_label_ids = torch.masked_fill(truncated_label_ids, truncated_label_ids==tokenizer.pad_token_id, -100)
            truncated_response_ids = torch.masked_fill(truncated_response_ids, truncated_response_ids==tokenizer.pad_token_id, -100)
            
            memorized_ids = (truncated_label_ids == truncated_response_ids)
            valid_ids = (truncated_label_ids != -100) & (truncated_response_ids != -100)
            all_memorized_ids.append(memorized_ids)
            all_valid_ids.append(valid_ids)
            

    all_memorized_ids = torch.cat(all_memorized_ids)
    all_valid_ids = torch.cat(all_valid_ids)
    all_memorized_ids = all_gather(all_memorized_ids)
    all_valid_ids = all_gather(all_valid_ids)

    return all_memorized_ids.cpu(), all_valid_ids.cpu()


def evaluate_memorization(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_memorization.pt")
    
    if not os.path.exists(save_path):
        all_memorized_ids, all_valid_ids = run_model_memorization(args, tokenizer, model, dataset, epoch, device)
        torch.save((all_memorized_ids, all_valid_ids), save_path) 
    else:
        all_memorized_ids, all_valid_ids = torch.load(save_path)
        print(f"rank {dist.get_rank()} all memorized size: {all_memorized_ids.size()}")
    return all_memorized_ids.numpy(), all_valid_ids.numpy()


def run_model_memorization_cat_label(args, tokenizer, model, dataset: PromptDataset, epoch, device, desc=None):
    
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    
    generation_config = GenerationConfig (
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=args.min_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    all_memorized_ids, all_valid_ids = [], []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=desc, disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)

            query_ids = model_batch["input_ids"]
            label_ids = no_model_batch["rest_ids"]
            # model_batch["input_ids"] = torch.cat([query_ids[:,16:], label_ids[:,:16]], dim=1)

            max_new_tokens = args.max_length - query_ids.size(1) 
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            
            truncated_label_ids = label_ids[:, :32]
            truncated_response_ids = response_ids[:, :32]
            truncated_label_ids = torch.masked_fill(truncated_label_ids, truncated_label_ids==tokenizer.pad_token_id, -100)
            truncated_response_ids = torch.masked_fill(truncated_response_ids, truncated_response_ids==tokenizer.pad_token_id, -100)
            
            memorized_ids = (truncated_label_ids == truncated_response_ids)
            valid_ids = (truncated_label_ids != -100) & (truncated_response_ids != -100)
            all_memorized_ids.append(memorized_ids)
            all_valid_ids.append(valid_ids)
            

    all_memorized_ids = torch.cat(all_memorized_ids)
    all_valid_ids = torch.cat(all_valid_ids)
    all_memorized_ids = all_gather(all_memorized_ids)
    all_valid_ids = all_gather(all_valid_ids)

    return all_memorized_ids.cpu(), all_valid_ids.cpu()


def evaluate_memorization_cat_label(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_memorization_cat_label.pt")
    
    if not os.path.exists(save_path):
        all_memorized_ids, all_valid_ids = run_model_memorization_cat_label(args, tokenizer, model, dataset, epoch, device)
        torch.save((all_memorized_ids, all_valid_ids), save_path) 
    else:
        all_memorized_ids, all_valid_ids = torch.load(save_path)
        print(f"rank {dist.get_rank()} all memorized size: {all_memorized_ids.size()}")
    return all_memorized_ids.numpy(), all_valid_ids.numpy()


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    # Assert generated length is fixed to 64
    # assert args.max_length == 64
    # assert args.min_length == 64
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    print("OK")
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"] 
    args.deepspeed_config = None

    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset_mi(
        args,
        tokenizer,
    )

    model = setup_model(args, ds_config, device)
    
    # cat_label_train_memorized_ids, cat_label_train_valid_ids = evaluate_memorization_cat_label(args, tokenizer, model, dataset["train"], "train", 0, device)
    # cat_label_train_memorized_acc = cat_label_train_memorized_ids.sum() / cat_label_train_valid_ids.sum()
    # cat_label_valid_memorized_ids, cat_label_valid_valid_ids = evaluate_memorization_cat_label(args, tokenizer, model, dataset["test"], "valid", 0, device)
    # cat_label_valid_memorized_acc = cat_label_valid_memorized_ids.sum() / cat_label_valid_valid_ids.sum()
    # print(f"Train memorized acc: {cat_label_train_memorized_acc}, valid memorized acc: {cat_label_valid_memorized_acc}")

    train_memorized_ids, train_valid_ids = evaluate_memorization(args, tokenizer, model, dataset["train"], "train", 0, device)
    # train_memorized_acc = train_memorized_ids.sum() / train_valid_ids.sum()
    train_memorized_acc = train_memorized_ids.mean().item()
    valid_memorized_ids, valid_valid_ids = evaluate_memorization(args, tokenizer, model, dataset["test"], "valid", 0, device)
    # valid_memorized_acc = valid_memorized_ids.sum() / valid_valid_ids.sum()
    valid_memorized_acc = valid_memorized_ids.mean().item()
    print(f"Train memorized acc: {train_memorized_acc}, valid memorized acc: {valid_memorized_acc}")
    memorized_acc_gap = train_memorized_acc - valid_memorized_acc
    save_path = os.path.join(args.save, "memorization_acc.json")
    with open(save_path, "w") as f:
        json.dump(
            {
                "train": train_memorized_acc, "valid": valid_memorized_acc,
                "gap": memorized_acc_gap,
                # "cat_label_train": cat_label_train_memorized_acc, "cat_label_valid": cat_label_valid_memorized_acc
            }, f, indent=4)
    

        
if __name__ == "__main__":
    main()