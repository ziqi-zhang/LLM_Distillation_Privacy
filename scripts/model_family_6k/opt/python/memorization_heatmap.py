import os, sys
import os.path as osp
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pdb import set_trace as st

model_name = "opt"
kd_names = ["kd", "seqkd", "gkd", "imitkd", "minillm", "distillm"]
mia_nams = ["loss", "mink", "minkpp", "ref", "ref_pretrain_teacher", "zlib"]
model_sizes = ["1.3B", "0.3B", "0.1B"]

kd_to_label = {
    "kd": "KD",
    "seqkd": "SeqKD",
    "gkd": "GKD",
    "imitkd": "ImitKD",
    "minillm": "MiniLLM",
    "distillm": "DistillM",
    "no_mem": "No Mem."
}
size_to_title = {
    "1.3B": "OPT 1.3B",
    "0.3B": "OPT 355M",
    "0.1B": "OPT 125M",
}

def load_train_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results["train"]

def load_gap_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    return abs(results["train"] - results["valid"])

results, gap_results = {}, {}
for size in model_sizes:
    results[size], gap_results[size] = {}, {}
    for kd in kd_names:
        results[size][kd] = load_train_results(f"results_6k/{model_name}/{size}/{kd}/memorization/memorization_acc.json")
        gap_results[size][kd] = load_gap_results(f"results_6k/{model_name}/{size}/{kd}/memorization/memorization_acc.json")
    
    results[size]["no_mem"] = load_train_results(f"results_6k/{model_name}/student_ref_sft/{size}/memorization/memorization_acc.json")
    gap_results[size]["no_mem"] = load_gap_results(f"results_6k/{model_name}/student_ref_sft/{size}/memorization/memorization_acc.json")
        

        
save_dir = f"results_6k/{model_name}/heatmap"
if not osp.exists(save_dir):
    os.makedirs(save_dir)
    
    
df = pd.DataFrame(results).T
# print(df)
plt.figure(figsize=(6,3))
sns.heatmap(
    df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True),
    xticklabels=[kd_to_label[kd] for kd in kd_names+["no_mem"]],
    yticklabels=[size_to_title[size] for size in model_sizes],
)  # annot=True 用于显示数值，cmap 设置颜色风格
# Rotate the tick labels for better visibility
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.title("Memorization Accuracy")
plt.tight_layout()
plt.savefig(f"{save_dir}/{model_name}_memorization_heatmap.png")
plt.clf()


df = pd.DataFrame(gap_results).T
# print(df)
plt.figure(figsize=(6,3))
sns.heatmap(
    df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True),
    xticklabels=[kd_to_label[kd] for kd in kd_names+["no_mem"]],
    yticklabels=[size_to_title[size] for size in model_sizes],
)  # annot=True 用于显示数值，cmap 设置颜色风格
# Rotate the tick labels for better visibility
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.title("Memorization Gap")
plt.tight_layout()
plt.savefig(f"{save_dir}/{model_name}_memorization_gap_heatmap.png")
plt.clf()
