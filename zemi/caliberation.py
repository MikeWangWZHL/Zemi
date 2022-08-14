import argparse
import logging
import os
import random
import json

import datasets
import torch
from datasets import (
    load_dataset,
    load_metric,
    load_from_disk,
    concatenate_datasets
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
)
from promptsource.templates import DatasetTemplates


def get_calibrate_matrices(batch, model):
    model_inputs = {
        "input_ids":batch["dummy_input_ids"],
        "attention_mask":batch["dummy_attention_mask"],
        "labels":batch["labels"], 
        "aug_input_ids":batch["aug_input_ids"], 
        "aug_attention_mask":batch["aug_attention_mask"]
    }
    with torch.no_grad():
        logits = model(**model_inputs).logits
    
    masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)       
    seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
    seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
    # This reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
    seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1) # b n_ans
    seq_log_prob = torch.softmax(seq_log_prob, dim=-1)
    # predictions = seq_log_prob.argmax(dim=-1)
    Ws = []
    for p_hat in seq_log_prob:
        w = torch.diag(p_hat)
        # w = torch.inverse(w) #TODO: solve singular problem
        Ws.append(w)
    # Ws = torch.stack(Ws)
    return Ws

def calibrate_probs(probs, batch, model):
    Diags = get_calibrate_matrices(batch,model)
    for i in range(probs.shape[0]):
        probs[i] = torch.linalg.lstsq(Diags[i],probs[i]).solution
        # probs[i] += torch.linalg.lstsq(Diags[i],probs[i]).solution
    return probs


def get_task_template_name(dir_path, dataset_path, kic_aug = None):
    dir_name = os.path.basename(dir_path)
    dataset_name = os.path.basename(dataset_path)
    if '__' in dir_name:
        d_name, d_conf_name = dir_name.split("__")[:2]
        if d_conf_name in ["None","none"]:
            task_full_name = d_name
        else:
            task_full_name = f"{d_name}_{d_conf_name}"
    else:
        task_full_name = dir_name
    if kic_aug is None:
        template_name = dataset_name.replace(f'{task_full_name}_', '').replace('_score_eval', '').strip('_')
    else:
        template_name = dataset_name.replace(f'{kic_aug}_', '').replace('_score_eval', '').strip('_')
    if 'anli' in task_full_name:
        template_name = template_name.replace('_r1', '').replace('_r2', '').replace('_r3', '')
    
    template_name = template_name.replace('-', '_').replace(' ', '_').replace('/', '_').replace('___', '_')
    template_name = re.sub(r"[^\w\d'\s\_]+", '', template_name).strip('_')
    return task_full_name, template_name


import re
from collections import defaultdict

def load_template_dict():
    res = defaultdict(dict)
    with open("/cephfs/user/mikeeewang/summer_22/workspace/data/p3_template_hf_dataset/template_dataset_v1.1.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            dataset_name, dataset_config_name = obj["dataset_name"], obj["dataset_config_name"]
            if dataset_config_name:
                task_full_name = f'{dataset_name}_{dataset_config_name}'
            else:
                task_full_name = f'{dataset_name}'
            tn = obj['template_name'].replace('-', '_').replace(' ', '_').replace('/', '_').replace('___', '_')
            tn = re.sub(r"[^\w\d'\s\_]+", '', tn).strip('_')
            res[task_full_name][tn] = obj['template_str']
    return res