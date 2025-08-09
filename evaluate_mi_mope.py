import time
import os
import os.path as osp

import torch
import torch.distributed as dist
import deepspeed
import copy
import shutil
from tqdm import tqdm
from functools import partial

import json
from pdb import set_trace as st
from transformers import mpu, AutoModelForCausalLM, AutoTokenizer

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model
from data_utils.prompt_datasets import PromptDataset
from mi_utils import compute_AUC, plot_ROC, compute_tpr_25_05_01_001_0

torch.set_num_threads(4)

from evaluate_mi import setup_model, prepare_dataset_mi, wrap_model
from evaluate_mi_main import (
    run_model_loss, run_model_mink, run_model_zlib
)
import evaluate_mi_main

perturbation_seeds, perturbation_model_paths = [], []


def noise_vector(size, noise_type):
    if noise_type == 1: # gaussian
        return torch.randn(size)
    elif noise_type == 2: # rademacher
        return torch.randint(0,2,size)*2-1
    else: # user-specified
        print(" - WARNING: noise_type not recognized. Using Gaussian noise. You can specify other options here.")
        return torch.randn(size)


def setup_perturbation_seeds(args):
    num_seeds = args.num_perturbation_models
    global perturbation_seeds
    # Generate random prime seeds
    perturbation_seeds = [i for i in range(1, num_seeds+1)]
    
@torch.no_grad()
def setup_perturbation_models(args, tokenizer, noise_type=1):
    raw_model = get_model(args, "cpu")
    noise_stdev = args.sigma
    perturbation_model_dir = osp.join(args.save, "perturbation_models")
    if dist.get_rank() == 0:
        if not osp.exists(perturbation_model_dir):
            os.makedirs(perturbation_model_dir)

    for idx, seed in enumerate(perturbation_seeds):
        
        seed_perturbation_model_dir = osp.join(perturbation_model_dir, f"seed_{seed}")
        if not osp.exists(seed_perturbation_model_dir):
            os.makedirs(seed_perturbation_model_dir)
        perturbation_model_paths.append(seed_perturbation_model_dir)
        
        if osp.exists(osp.join(seed_perturbation_model_dir, "model.safetensors")):
            log_str = f"Model for seed {seed} already exists in {seed_perturbation_model_dir}. Skipping..."
            print_rank(log_str, os.path.join(args.save, "log.txt"))
            continue
            
        log_str = f"Setting up perturbation model for seed {seed} ({idx+1}/{len(perturbation_seeds)})... "
        print(log_str)
        print_rank(log_str, os.path.join(args.save, "log.txt"))
            
        torch.manual_seed(seed)
        model = copy.deepcopy(raw_model)
        ## Perturbed model
        for name, param in model.named_parameters():
            noise = noise_vector(param.size(), noise_type) * noise_stdev
            param.add_(noise)
        
        model.save_pretrained(seed_perturbation_model_dir, from_pt=True)
        tokenizer.save_pretrained(seed_perturbation_model_dir)
        
        del model, name, param
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def remove_perturbation_models(args):
    perturbation_model_dir = osp.join(args.save, "perturbation_models")
    for seed in perturbation_seeds:
        seed_perturbation_model_dir = osp.join(perturbation_model_dir, f"seed_{seed}")
        if osp.exists(seed_perturbation_model_dir):
            shutil.rmtree(seed_perturbation_model_dir)
        
@torch.no_grad()
def evaluate_one_model(args, tokenizer, model, dataset, ds_config, desc):
    model = wrap_model(args, model, ds_config)
    
    losses = []
    for batch_idx, batch in tqdm(enumerate(dataset)):
        loss = model(**batch).loss
        losses.append(loss)
    
    losses = torch.stack(losses)
    return losses
    ...
    
def evaluate_mi_main_one_model(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)

    all_lm_losses, mean_lm_loss = run_model(args, tokenizer, model, dataset, epoch, device)

    return all_lm_losses

def get_statistics_mope(training_res, validation_res):
    
    train_diff = training_res[0,:]-training_res[1:,:].mean(dim=0)
    valid_diff = validation_res[0,:]-validation_res[1:,:].mean(dim=0)

    return train_diff, valid_diff

def get_statistics_mope_ratio(training_res, validation_res):
    train_diff = (training_res[0,:]-training_res[1:,:].mean(dim=0)) / training_res[0,:]
    valid_diff = (validation_res[0,:]-validation_res[1:,:].mean(dim=0)) / validation_res[0,:]

    return train_diff, valid_diff
    
def evaluate_mope_main(
    args, tokenizer, mia_fn_train, mia_fn_val, mia_name, dataset, ds_config, device,
):
    print(f"MIA Name: {mia_name}")
    model = get_model(args, device)
    model = wrap_model(args, model, ds_config)
    loss_save_dir = osp.join(args.save, mia_name)
    if not osp.exists(loss_save_dir) and dist.get_rank() == 0:
        os.makedirs(loss_save_dir)
    
    
    
    save_path = osp.join(loss_save_dir, f"original_{len(dataset['train'])}.pt")
    if not osp.exists(save_path):
        train_original = mia_fn_train(model=model, desc=f"{mia_name} original model on training data...")
        validation_original = mia_fn_val(model=model, desc=f"{mia_name} original model on validation data...")
        if dist.get_rank() == 0:
            torch.save((train_original, validation_original), save_path)
    else:
        train_original, validation_original = torch.load(save_path)

    training_res = torch.zeros((args.num_perturbation_models + 1, train_original.size(0)))  
    validation_res = torch.zeros((args.num_perturbation_models + 1, validation_original.size(0)))  
    training_res[0] = train_original
    validation_res[0] = validation_original
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize() 
    
    for idx, seed in enumerate(perturbation_seeds):
        
        save_path = osp.join(loss_save_dir, f"seed_{seed}_{len(dataset['train'])}.pt")
        if not osp.exists(save_path):
            seed_perturbation_model_dir = perturbation_model_paths[idx]
            perturb_model = AutoModelForCausalLM.from_pretrained(seed_perturbation_model_dir)
            model = wrap_model(args, perturb_model, ds_config)
            
            train_save_path = osp.join(loss_save_dir, f"seed_{seed}_train.pt")
            if os.path.exists(train_save_path):
                train_loss = torch.load(train_save_path)
            else:
                train_loss = mia_fn_train(model=model, desc=f"model {idx+1}/{len(perturbation_seeds)} on training {loss_save_dir}")
                if dist.get_rank() == 0:
                    torch.save(train_loss, train_save_path)
            
            valid_save_path = osp.join(loss_save_dir, f"seed_{seed}_validation.pt")
            if os.path.exists(valid_save_path):
                validation_loss = torch.load(valid_save_path)
            else:
                validation_loss = mia_fn_val(model=model, desc=f"model {idx+1}/{len(perturbation_seeds)} on validation {loss_save_dir}")
                if dist.get_rank() == 0:
                    torch.save(validation_loss, valid_save_path)
            if dist.get_rank() == 0:
                torch.save((train_loss, validation_loss), save_path)
                # remove training and validation losses
                os.remove(train_save_path)
                os.remove(valid_save_path)
            
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            train_loss, validation_loss = torch.load(save_path)
        training_res[idx+1] = train_loss
        validation_res[idx+1] = validation_loss
        
    return training_res, validation_res

def check_all_mia_results(args):
    # for mia_name, _, _ in mia_name_fn_pairs:
    for mia_name in ["loss"]:
        save_path = os.path.join(args.save, f"mope_results_{mia_name}.json")
        if not os.path.exists(save_path):
            return False
    return True
    
    
def evaluate_all_mi(args, tokenizer, dataset, ds_config, device):
    
    # Reference attack
    ref_args = copy.deepcopy(args)
    ref_args.model_path = "stabilityai/stablelm-base-alpha-3b-v2"
    ref_model = get_model(ref_args, device)
    ref_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
    ref_tokenizer.pad_token_id = ref_tokenizer.eos_token_id
    
    ref_dataset = prepare_dataset_mi(
        args,
        ref_tokenizer,
    )
    print(tokenizer.decode(dataset["train"][0][2]))
    print(ref_tokenizer.decode(ref_dataset["train"][0][2]))
    
    mia_name_fn_pairs = [
        (
            "loss", 
            partial(evaluate_mi_main.run_model_loss, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device,),
            partial(evaluate_mi_main.run_model_loss, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device,),
        ),
        # (
        #     "mink", 
        #     partial(evaluate_mi_main.run_model_mink, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device,),
        #     partial(evaluate_mi_main.run_model_mink, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device,),
        # ),
        # (
        #     "minkpp",
        #     partial(evaluate_mi_main.run_model_mink, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device, minkpp=True),
        #     partial(evaluate_mi_main.run_model_mink, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device, minkpp=True),
        # ),
        # (
        #     "zlib",
        #     partial(evaluate_mi_main.run_model_zlib, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device,),
        #     partial(evaluate_mi_main.run_model_zlib, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device,),
        # ),
    ]
    

    for mia_name, mia_fn_train, mia_fn_val in mia_name_fn_pairs:
        training_res, validation_res = evaluate_mope_main(
            args, tokenizer, mia_fn_train, mia_fn_val, mia_name, dataset, ds_config, device
        )
    
        if dist.get_rank() == 0:
            compute_and_save_results(training_res, validation_res, mia_name, args)

            
def evaluate_all_mi_ref_attack(args, tokenizer, dataset, ds_config, device):
    
    # Reference attack
    ref_args = copy.deepcopy(args)
    ref_args.model_path = "stabilityai/stablelm-base-alpha-3b-v2"
    ref_model = get_model(ref_args, device)
    ref_tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
    ref_tokenizer.pad_token_id = ref_tokenizer.eos_token_id
    
    ref_dataset = prepare_dataset_mi(
        args,
        ref_tokenizer,
    )
    print(tokenizer.decode(dataset["train"][0][2]))
    print(ref_tokenizer.decode(ref_dataset["train"][0][2]))
    
    mia_name = "ref"
    mia_fn_train = partial(
        evaluate_mi_main.run_model_ref, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device, 
        ref_tokenizer=ref_tokenizer, ref_model=ref_model, ref_dataset=ref_dataset["train"]
    )
    mia_fn_val = partial(
        evaluate_mi_main.run_model_ref, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device, 
        ref_tokenizer=ref_tokenizer, ref_model=ref_model, ref_dataset=ref_dataset["test"]
    )
        
    training_res, validation_res = evaluate_mope_main(
        args, tokenizer, mia_fn_train, mia_fn_val, mia_name, dataset, ds_config, device
    )
    if dist.get_rank() == 0:
        compute_and_save_results(training_res, validation_res, mia_name, args)
        
    ref_model = ref_model.cpu()
    del ref_model
    # clear GPU memory
    torch.cuda.empty_cache()

    
def evaluate_all_mi_ref_same_family_attack(args, tokenizer, dataset, ds_config, device):
    # Reference attack
    ref_args = copy.deepcopy(args)
    if args.model_type == "gpt2":
        ref_args.model_path = "gpt2-xl"
    elif args.model_type == "opt":
        ref_args.model_path = "facebook/opt-2.7B"
    else:
        raise not NotImplementedError
    ref_model = get_model(ref_args, device)
    
    mia_name = "ref_same_family"
    mia_fn_train = partial(
        evaluate_mi_main.run_model_ref_same_family, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device,
        ref_model=ref_model,
    )
    mia_fn_val = partial(
        evaluate_mi_main.run_model_ref_same_family, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device, 
        ref_model=ref_model,
    )   
    training_res, validation_res = evaluate_mope_main(
        args, tokenizer, mia_fn_train, mia_fn_val, mia_name, dataset, ds_config, device
    )
    if dist.get_rank() == 0:
        compute_and_save_results(training_res, validation_res, mia_name, args)
        
    ref_model = ref_model.cpu()
    del ref_model
    # clear GPU memory
    torch.cuda.empty_cache()

def evaluate_all_mi_ref_same_domain_calibration(args, tokenizer, dataset, ds_config, device):
    # Reference attack
    ref_args = copy.deepcopy(args)
    ref_args.model_path = args.same_domain_calibration_path
    ref_model = get_model(ref_args, device)
    
    mia_name = "ref_same_domain"
    mia_fn_train = partial(
        evaluate_mi_main.run_model_ref_same_family, args=args, tokenizer=tokenizer, dataset=dataset["train"], epoch=0, device=device,
        ref_model=ref_model,
    )
    mia_fn_val = partial(
        evaluate_mi_main.run_model_ref_same_family, args=args, tokenizer=tokenizer, dataset=dataset["test"], epoch=0, device=device, 
        ref_model=ref_model,
    )   
    training_res, validation_res = evaluate_mope_main(
        args, tokenizer, mia_fn_train, mia_fn_val, mia_name, dataset, ds_config, device
    )
    if dist.get_rank() == 0:
        compute_and_save_results(training_res, validation_res, mia_name, args)
        
    ref_model = ref_model.cpu()
    del ref_model
    # clear GPU memory
    torch.cuda.empty_cache()

def compute_and_save_results(training_res, validation_res, mia_name, args):
    train_metrics, val_metrics = get_statistics_mope(training_res, validation_res)
    AUC = compute_AUC(train_metrics, val_metrics)
    print(f"{mia_name} AUC: ", AUC)
    tpr_25, tpr_05, tpr_01, tpr_001, tpr_0 = compute_tpr_25_05_01_001_0(train_metrics, val_metrics)
    plot_ROC(
        train_metrics, val_metrics, title="ROC", log_scale=False, show_plot=True, 
        save_name=os.path.join(args.save, f"ROC_mope_{mia_name}.png")
    )
    plot_ROC(
        train_metrics, val_metrics, title="ROC", log_scale=True, show_plot=True, 
        save_name=os.path.join(args.save, f"ROC_log_mope_{mia_name}.png")
    )
    results = {
        "AUC": AUC,
        "tpr_25": tpr_25,
        "tpr_05": tpr_05,
        "tpr_01": tpr_01,
        "tpr_001": tpr_001,
        "tpr_0": tpr_0,
    }
    save_path = os.path.join(args.save, f"mope_results_{mia_name}.json")
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
    if args.type == "eval_main" or args.type == "mope":
        dataset = prepare_dataset_mi(
            args,
            tokenizer,
        )
    else:
        raise NotImplementedError
    # model = setup_model(args, ds_config, device)
    
    if check_all_mia_results(args):
        print("All MIA results already exist. Skipping...")
        return
    
    setup_perturbation_seeds(args)
    setup_perturbation_models(args, tokenizer)

    evaluate_all_mi(args, tokenizer, dataset, ds_config, device)
    # evaluate_all_mi_ref_attack(args, tokenizer, dataset, ds_config, device)
    # evaluate_all_mi_ref_same_family_attack(args, tokenizer, dataset, ds_config, device)
    
    if dist.get_rank() == 0:
        remove_perturbation_models(args)
    
    
    
if __name__ == "__main__":
    main()