import time
import os

import torch
import torch.distributed as dist
import deepspeed

import json
from pdb import set_trace as st
from transformers import mpu

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model
from data_utils.prompt_datasets import PromptDataset
from evaluate_mi_main import evaluate_mi_main, prepare_dataset_main
from evaluate_exposure_bias import evaluate_eb, prepare_dataset_eb
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

    return data


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
    if args.type == "eval_main":
        dataset = prepare_dataset_mi(
            args,
            tokenizer,
        )
    elif args.type == "eval_exposure_bias":
        dataset = prepare_dataset_eb(
            args,
            tokenizer,
        )
    else:
        raise NotImplementedError
    model = setup_model(args, ds_config, device)
    
    if args.type == "eval_main":
        train_metrics = evaluate_mi_main(args, tokenizer, model, dataset["train"], "train", 0, device)
        val_metrics = evaluate_mi_main(args, tokenizer, model, dataset["test"], "test", 0, device)
    elif args.type == "eval_exposure_bias":
        evaluate_eb(args, tokenizer, model, dataset["test"], "test", 0, device)
    else:
        raise NotImplementedError
    
    print("train metric shape: ", train_metrics.shape)
    print("val metric shape: ", val_metrics.shape)
    
    AUC = compute_AUC(train_metrics, val_metrics)
    tpr_25, tpr_05, tpr_01, tpr_001, tpr_0 = compute_tpr_25_05_01_001_0(train_metrics, val_metrics)
    print("AUC: ", AUC)
    plot_ROC(
        train_metrics, val_metrics, title="ROC", log_scale=False, show_plot=True, 
        save_name=os.path.join(args.save, "loss_ROC.png")
    )
    
    results = {
        "AUC": AUC,
        "tpr_25": tpr_25,
        "tpr_05": tpr_05,
        "tpr_01": tpr_01,
        "tpr_001": tpr_001,
        "tpr_0": tpr_0,
    }
    save_path = os.path.join(args.save, "results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    
    
if __name__ == "__main__":
    main()