import torch.distributed
from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig, mpu

import os
import nltk
nltk.download("punkt")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, all_gather
import zlib

from pdb import set_trace as st

from rouge_metric import compute_metrics

torch.set_num_threads(4)

def prepare_dataset_main(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data


def run_model_loss(args, tokenizer, model, dataset: PromptDataset, epoch, device, desc=None):
    
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

    all_lm_losses = []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=desc, disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            if args.model_parallel:
                lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
                lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
                raise NotImplementedError
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
            all_lm_losses.append(lm_loss)

    # print(f"rank {dist.get_rank()} all_lm_losses size: {len(all_lm_losses)}")
    all_lm_losses = torch.cat(all_lm_losses)
    all_lm_losses = all_gather(all_lm_losses)
    return all_lm_losses.cpu()

def evaluate_mi_loss(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_lm_losses.pt")
    
    if not os.path.exists(save_path):
        all_lm_losses = run_model_loss(args, tokenizer, model, dataset, epoch, device)
        torch.save(all_lm_losses, save_path) 
    else:
        all_lm_losses = torch.load(save_path)
        print(f"rank {dist.get_rank()} lm loss size: {all_lm_losses.size()}")
    return all_lm_losses.numpy()



def run_model_mink(args, tokenizer, model, dataset: PromptDataset, epoch, device, minkpp=False, desc=None):
    
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()

    all_lm_losses = []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=desc, disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            # print_rank("input_id size: ", model_batch["input_ids"].size(), "rest_ids size: ", no_model_batch["rest_ids"].size(), "all_id size: ", all_ids.size())
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            confidence = torch.nn.functional.softmax(logits, dim=-1)
            log_confidence = torch.nn.functional.log_softmax(logits, dim=-1)
            # token_confidence  = confidence.max(-1)[0]
            mink_scores = []
            for i in range(confidence.size(0)):
                valid_sample_confidence = confidence[i][loss_mask[i] == 1].float()
                valid_sample_log_confidence = log_confidence[i][loss_mask[i] == 1].float()
                valid_label_ids = label_ids[i][loss_mask[i] == 1].long()
                label_conf = []
                for j in range(valid_sample_confidence.size(0)):
                    target_conf = valid_sample_confidence[j][valid_label_ids[j]]
                    label_conf.append(target_conf)
                label_conf = torch.stack(label_conf)
                if minkpp:
                    mu = (valid_sample_log_confidence * valid_sample_confidence).sum(-1)
                    sigma = (valid_sample_confidence * torch.square(valid_sample_log_confidence)).sum(-1) - torch.square(mu) 
                    norm_score = (label_conf - mu) / sigma.sqrt()
                    mia_score = norm_score
                else:
                    mia_score = label_conf
                sample_confidence_sorted, indices = torch.sort(mia_score, descending=False)
                mink_conf = sample_confidence_sorted[:int(0.2 * mia_score.size(0))]
                mink_scores.append(-mink_conf.mean())
                # print(f"rank {dist.get_rank()} mink score: {mink_scores[-1]}")
            mink_scores = torch.stack(mink_scores)

            all_lm_losses.append(mink_scores)

    all_lm_losses = torch.cat(all_lm_losses)
    all_lm_losses = all_gather(all_lm_losses)
        
    return all_lm_losses.cpu()

def evaluate_mi_mink(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_lm_mink.pt")
    
    if not os.path.exists(save_path):
        all_lm_losses = run_model_mink(args, tokenizer, model, dataset, epoch, device)
        torch.save(all_lm_losses, save_path) 
    else:
        all_lm_losses = torch.load(save_path)
        print(f"rank {dist.get_rank()} lm loss size: {all_lm_losses.size()}")
    return all_lm_losses.numpy()
        
def evaluate_mi_minkpp(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_lm_minkpp.pt")
    
    if not os.path.exists(save_path):
        all_lm_losses = run_model_mink(args, tokenizer, model, dataset, epoch, device, minkpp=True)
        torch.save(all_lm_losses, save_path) 
    else:
        all_lm_losses = torch.load(save_path)
        print(f"rank {dist.get_rank()} lm loss size: {all_lm_losses.size()}")
    return all_lm_losses.numpy()

    

def run_model_zlib(args, tokenizer, model, dataset: PromptDataset, epoch, device, desc=None):
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()

    all_lm_losses = []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=desc, disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            if args.model_parallel:
                lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
                # lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                # lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
                
            confidence = torch.nn.functional.softmax(logits, dim=-1)
            token_confidence, token_indexes  = confidence.max(-1)
            zlib_scores = []
            for i in range(confidence.size(0)):
                valid_token_indexes = token_indexes[i][loss_mask[i] == 1]
                valid_losses = lm_loss[i][loss_mask[i] == 1]
                valid_mean_loss = valid_losses.mean()

                text = tokenizer.decode(valid_token_indexes)
                zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))
                score = valid_mean_loss / zlib_entropy
                zlib_scores.append(score)

            zlib_scores = torch.stack(zlib_scores)
            all_lm_losses.append(zlib_scores)

    all_lm_losses = torch.cat(all_lm_losses)
    all_lm_losses = all_gather(all_lm_losses)
        
    return all_lm_losses.cpu()

def evaluate_mi_zlib(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_lm_zlib.pt")
    
    if not os.path.exists(save_path):
        all_lm_losses = run_model_zlib(args, tokenizer, model, dataset, epoch, device, )
        torch.save(all_lm_losses, save_path) 
    else:
        all_lm_losses = torch.load(save_path)
    return all_lm_losses.numpy()



def run_model_ref(
    args, 
    tokenizer, model, dataset: PromptDataset, 
    epoch, device, 
    ref_tokenizer, ref_model, ref_dataset,
    desc=None,
):
    collate_fn = dataset.collate
    ref_collate_fn = ref_dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    ref_sampler = DistributedSampler(ref_dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    ref_dataloader = DataLoader(
        ref_dataset, sampler=ref_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=ref_collate_fn)
    model.eval()
    ref_model.eval()

    all_lm_losses = []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (
            (model_batch, no_model_batch), (ref_model_batch, ref_no_model_batch)
        ) in enumerate(tqdm(zip(dataloader, ref_dataloader), desc=desc, disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)
            ref_dataset.move_to_device(ref_model_batch, ref_no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            if args.model_parallel:
                raise notImplementedError
                lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
                lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
                 
            # Same for ref model
            ref_all_ids = torch.cat([ref_model_batch["input_ids"], ref_no_model_batch["rest_ids"]], dim=-1)
            ref_input_ids = ref_all_ids[:, :-1]
            ref_attention_mask = (ref_input_ids != ref_tokenizer.pad_token_id).long()
            ref_label_ids = ref_all_ids[:, 1:]
            ref_label_ids = torch.masked_fill(ref_label_ids, ref_label_ids==ref_tokenizer.pad_token_id, -100)
            ref_label_ids[:, :ref_model_batch["input_ids"].size(1)-1] = -100
            # for reference attack, we use stabilityai/stablelm-base-alpha-3b-v2, which is not gpt2, it is gpt-neo
            ref_out = ref_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask, return_dict=True)
            ref_logits = ref_out.logits
            ref_loss_mask = (ref_label_ids != -100).float()
            if args.model_parallel:
                raise notImplementedError
                lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
                lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                ref_lm_loss = loss_func(ref_logits.view(-1, ref_logits.size(-1)), ref_label_ids.view(-1)).view(ref_label_ids.size())
                ref_lm_loss = torch.sum(ref_lm_loss * ref_loss_mask, -1) / torch.sum(ref_loss_mask, -1)

            # print(tokenizer.decode(label_ids[0][label_ids[0]!=-100]))
            # print(ref_tokenizer.decode(ref_label_ids[0][ref_label_ids[0]!=-100]))
            # if not (tokenizer.decode(label_ids[0][label_ids[0]!=-100][:10]) == ref_tokenizer.decode(ref_label_ids[0][ref_label_ids[0]!=-100][:10])):
            #     print_rank(f"rank {dist.get_rank()} label mismatch")
            #     print_rank(tokenizer.decode(label_ids[0][label_ids[0]!=-100][:10]))
            #     print_rank(ref_tokenizer.decode(ref_label_ids[0][ref_label_ids[0]!=-100][:10]))
            # assert (tokenizer.decode(label_ids[0][label_ids[0]!=-100][:10]) == ref_tokenizer.decode(ref_label_ids[0][ref_label_ids[0]!=-100][:10]))
            # GPT2's tokenizer and stablelm's tokenizer are different for some space/punctuation tokens, so can not directly compare
            
            
            lm_loss = lm_loss - ref_lm_loss
            all_lm_losses.append(lm_loss)

    all_lm_losses = torch.cat(all_lm_losses)
    all_lm_losses = all_gather(all_lm_losses)
        
    return all_lm_losses.cpu()

def evaluate_mi_ref(
    args, 
    tokenizer, model, dataset: PromptDataset, 
    ref_tokenizer, ref_model, ref_dataset,
    split, epoch, device
):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_lm_ref.pt")
    if not os.path.exists(save_path):
        all_lm_losses = run_model_ref(
            args, 
            tokenizer, model, dataset, 
            epoch, device,
            ref_tokenizer, ref_model, ref_dataset,
        )
        torch.save(all_lm_losses, save_path) 
    else:
        all_lm_losses = torch.load(save_path)
        print(f"rank {dist.get_rank()} lm loss size: {all_lm_losses.size()}")
    return all_lm_losses.numpy()
        

        

def run_model_ref_same_family(args, tokenizer, model, ref_model, dataset: PromptDataset, epoch, device, desc=None):
    
    collate_fn = dataset.collate
    
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    ref_model.eval()

    all_lm_losses = []

    if desc is None:
        desc = f"Evaluating {args.data_names} "

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=desc, disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            # print_rank("input_id size: ", model_batch["input_ids"].size(), "rest_ids size: ", no_model_batch["rest_ids"].size(), "all_id size: ", all_ids.size())
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            
            
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            if args.model_parallel:
                lm_loss = mpu.parallel_cross_entropy(logits, label_ids)
                lm_loss = torch.sum(lm_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
                raise NotImplementedError
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
                
            
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                ref_out = ref_model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ref_logits = ref_out.logits
            ref_loss_mask = (label_ids != -100).float()
            if args.model_parallel:
                ref_lm_loss = mpu.parallel_cross_entropy(ref_logits, label_ids)
                ref_lm_loss = torch.sum(ref_lm_loss * ref_loss_mask, dim=-1) / torch.sum(ref_loss_mask, dim=-1)
                raise NotImplementedError
            else:
                loss_func = nn.CrossEntropyLoss(reduction="none")
                ref_lm_loss = loss_func(ref_logits.view(-1, ref_logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
                ref_lm_loss = torch.sum(ref_lm_loss * ref_loss_mask, -1) / torch.sum(ref_loss_mask, -1)
                
            lm_loss = lm_loss - ref_lm_loss
            
            all_lm_losses.append(lm_loss)

    all_lm_losses = torch.cat(all_lm_losses)
    all_lm_losses = all_gather(all_lm_losses)
    return all_lm_losses.cpu()

def evaluate_mi_ref_same_family(args, tokenizer, model, ref_model, dataset: PromptDataset, split, epoch, device, tag):
    len_dataset = len(dataset)
    
    save_path = os.path.join(args.save, f"{split}_{len_dataset}_lm_{tag}.pt")
    
    if not os.path.exists(save_path):
        all_lm_losses = run_model_ref_same_family(args, tokenizer, model, ref_model, dataset, epoch, device)
        torch.save(all_lm_losses, save_path) 
    else:
        all_lm_losses = torch.load(save_path)
        print(f"rank {dist.get_rank()} lm loss size: {all_lm_losses.size()}")
    return all_lm_losses.numpy()
