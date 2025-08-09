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

def compute_save_metrics(train_metrics, val_metrics, name, args):
    print("train metric shape: ", train_metrics.shape)
    print("val metric shape: ", val_metrics.shape)
    
    num_samples = train_metrics.shape[0]
    AUC = compute_AUC(train_metrics, val_metrics)
    tpr_25, tpr_05, tpr_01, tpr_001, tpr_0 = compute_tpr_25_05_01_001_0(train_metrics, val_metrics)
    print("AUC: ", AUC)
    plot_ROC(
        train_metrics, val_metrics, title="ROC", log_scale=False, show_plot=True, 
        save_name=os.path.join(args.save, f"ROC_{name}.png")
    )
    
    results = {
        "num_samples": num_samples,
        "AUC": AUC,
        "tpr_25": tpr_25,
        "tpr_05": tpr_05,
        "tpr_01": tpr_01,
        "tpr_001": tpr_001,
        "tpr_0": tpr_0,
    }
    save_path = os.path.join(args.save, f"results_{name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
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

    # Loss attack
    loss_train_metrics = evaluate_mi_loss(args, tokenizer, model, dataset["train"], "train", 0, device)
    loss_val_metrics = evaluate_mi_loss(args, tokenizer, model, dataset["test"], "test", 0, device)
    if dist.get_rank() == 0:
        compute_save_metrics(loss_train_metrics, loss_val_metrics, "loss", args)
    
    # Mink Attack
    mink_train_metrics = evaluate_mi_mink(args, tokenizer, model, dataset["train"], "train", 0, device)
    mink_val_metrics = evaluate_mi_mink(args, tokenizer, model, dataset["test"], "test", 0, device)
    if dist.get_rank() == 0:
        compute_save_metrics(mink_train_metrics, mink_val_metrics, "mink", args)
        
    # Minkpp Attack
    minkpp_train_metrics = evaluate_mi_minkpp(args, tokenizer, model, dataset["train"], "train", 0, device)
    minkpp_val_metrics = evaluate_mi_minkpp(args, tokenizer, model, dataset["test"], "test", 0, device)
    if dist.get_rank() == 0:
        compute_save_metrics(minkpp_train_metrics, minkpp_val_metrics, "minkpp", args)
    
    # Zlib Attack
    zlib_train_metrics = evaluate_mi_zlib(args, tokenizer, model, dataset["train"], "train", 0, device)
    zlib_val_metrics = evaluate_mi_zlib(args, tokenizer, model, dataset["test"], "test", 0, device)
    if dist.get_rank() == 0:
        compute_save_metrics(zlib_train_metrics, zlib_val_metrics, "zlib", args)

    # Reference attack
    ref_args = copy.deepcopy(args)
    ref_args.model_path = "stabilityai/stablelm-base-alpha-3b-v2"
    ref_args.peft = None
    ref_model = get_model(ref_args, device)
    ref_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
    ref_tokenizer.pad_token_id = ref_tokenizer.eos_token_id
    
    ref_dataset = prepare_dataset_mi(
        args,
        ref_tokenizer,
    )
    tokenizer.decode(dataset["train"][0][2])
    ref_tokenizer.decode(ref_dataset["train"][0][2])
    ref_train_metrics = evaluate_mi_ref(
        args, 
        tokenizer, model, dataset["train"], 
        ref_tokenizer, ref_model, ref_dataset["train"],
        "train", 0, device
    )
    ref_val_metrics = evaluate_mi_ref(
        args, 
        tokenizer, model, dataset["test"], 
        ref_tokenizer, ref_model, ref_dataset["test"],
        "test", 0, device
    )
    if dist.get_rank() == 0:
        compute_save_metrics(ref_train_metrics, ref_val_metrics, "ref", args)
    ref_model = ref_model.cpu()
    del ref_model
    # clear GPU memory
    torch.cuda.empty_cache()


    # Reference attack using pretrained teacher model
    ref_args = copy.deepcopy(args)
    if args.model_type == "gpt2":
        ref_args.model_path = "gpt2-xl"
    elif args.model_type == "opt":
        ref_args.model_path = "facebook/opt-2.7B"
    elif args.model_type == "llama":
        ref_args.model_path = "openlm-research/open_llama_3b"
    else:
        raise not NotImplementedError
    ref_args.peft = None
    ref_model = get_model(ref_args, device)
    ref_train_metrics = evaluate_mi_ref_same_family(args, tokenizer, model, ref_model, dataset["train"], "train", 0, device, "ref_pretrain_teacher")
    ref_val_metrics = evaluate_mi_ref_same_family(args, tokenizer, model, ref_model, dataset["test"], "test", 0, device, "ref_pretrain_teacher")
    if dist.get_rank() == 0:
        compute_save_metrics(ref_train_metrics, ref_val_metrics, "ref_pretrain_teacher", args)
    ref_model = ref_model.cpu()
    del ref_model
    # clear GPU memory
    torch.cuda.empty_cache()

    # Reference attack using a model trained from the same domain
    if args.same_domain_calibration_path is not None and os.path.exists(args.same_domain_calibration_path):
        ref_args = copy.deepcopy(args)
        ref_args.model_path = args.same_domain_calibration_path
        ref_model = get_model(ref_args, device)
        ref_train_metrics = evaluate_mi_ref_same_family(args, tokenizer, model, ref_model, dataset["train"], "train", 0, device, "ref_same_domain")
        ref_val_metrics = evaluate_mi_ref_same_family(args, tokenizer, model, ref_model, dataset["test"], "test", 0, device, "ref_same_domain")
        if dist.get_rank() == 0:
            compute_save_metrics(ref_train_metrics, ref_val_metrics, "ref_same_domain", args)
        ref_model = ref_model.cpu()
        del ref_model
        # clear GPU memory
        torch.cuda.empty_cache()
        
    
if __name__ == "__main__":
    main()