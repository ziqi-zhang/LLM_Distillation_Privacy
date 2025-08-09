import os, sys
import os.path as osp
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pdb import set_trace as st

model_name = "gpt2"
kd_names = ["kd", "seqkd", "gkd", "imitkd", "minillm", "distillm"]
mia_nams = ["loss", "mink", "minkpp", "ref", "ref_pretrain_teacher", "zlib"]
model_sizes = ["large", "medium", "base"]
teacher_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]


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
    "large": "GPT-2 Large",
    "medium": "GPT-2 Medium",
    "base": "GPT-2 Base",
}

def load_train_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results["student_ratio_on_teacher_ratio"]

min_value, max_value = 1, 0

results, gap_results = {}, {}
for size in model_sizes:
    results[size], gap_results[size] = {}, {}
    for kd in kd_names:
        results[size][kd] = load_train_results(f"results_6k/{model_name}/{size}/{kd}/memorization_w_teacher/memorization_acc.json")
        min_value = min(min_value, min(results[size][kd].values()))
        max_value = max(max_value, max(results[size][kd].values()))

    
    results[size]["no_mem"] = load_train_results(f"results_6k/{model_name}/student_ref_sft/{size}/memorization_w_teacher/memorization_acc.json")
    min_value = min(min_value, min(results[size]["no_mem"].values()))
    max_value = max(max_value, max(results[size]["no_mem"].values()))

print(min_value, max_value)


        
save_dir = f"results_6k/{model_name}/heatmap/mem_w_teacher"
if not osp.exists(save_dir):
    os.makedirs(save_dir)
    

for size in model_sizes:

    df = pd.DataFrame(results[size])
    # print(df)
    plt.figure(figsize=(6,3))
    sns.heatmap(
        df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True),
        xticklabels=[kd_to_label[kd] for kd in kd_names+["no_mem"]],
        # yticklabels=[size_to_title[size] for size in model_sizes],
        # add min max value
        vmin=min_value, vmax=max_value
    )  # annot=True 用于显示数值，cmap 设置颜色风格
    # Rotate the tick labels for better visibility
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.title("Memorization Accuracy")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_{size}.png")
    plt.clf()
