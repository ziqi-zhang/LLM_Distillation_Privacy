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
# kd_names = ["kd", "gkd", "imitkd", "minillm", "distillm"]
mia_nams = ["loss", "mink", "minkpp", "ref", "ref_pretrain_teacher", "zlib"]
model_sizes = ["1.3B", "0.3B", "0.1B"]

mia_to_label = {
    "loss": "Loss",
    "mink": "Min-k%",
    "minkpp": "Min-k%++",
    "ref": "REF (STABLELM)",
    "ref_pretrain_teacher": "REF (Pretrain)",
    "zlib": "Zlib",
    "mope": "MoPe"
}
kd_to_label = {
    "kd": "KD",
    "seqkd": "SeqKD",
    "gkd": "GKD",
    "imitkd": "ImitKD",
    "minillm": "MiniLLM",
    "distillm": "DistillM",
}
size_to_title = {
    "1.3B": "OPT 1.3B",
    "0.3B": "OPT 355M",
    "0.1B": "OPT 125M",
}

def load_auc_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results["AUC"]

min_value, max_value = 1, 0
            
results = {}
for size in model_sizes:
    results[size] = {}
    for kd in kd_names:
        results[size][kd] = {}
        for mia in mia_nams:
            results[size][kd][mia] = load_auc_results(f"results_6k/{model_name}/{size}/{kd}/blackbox_mia/results_{mia}.json")
        min_value = min(min_value, min(results[size][kd].values()))
        max_value = max(max_value, max(results[size][kd].values()))

for size in model_sizes:
    for kd in kd_names:
        results[size][kd]["mope"] = load_auc_results(f"results_6k/{model_name}/{size}/{kd}/mope/sigma_0.005_models_10/mope_results_loss.json")
        min_value = min(min_value, results[size][kd]["mope"])
        max_value = max(max_value, results[size][kd]["mope"])

mia_nams.append("mope")

save_dir = f"results_6k/{model_name}/heatmap/mia"
if not osp.exists(save_dir):
    os.makedirs(save_dir)

for size in model_sizes:
    df = pd.DataFrame(results[size]).T
    # print(df)
    plt.figure(figsize=(6,4))
    sns.heatmap(
        df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True),
        xticklabels=[mia_to_label[mia] for mia in mia_nams],
        yticklabels=[kd_to_label[kd] for kd in kd_names],
        vmin=min_value, vmax=max_value,
    )  # annot=True 用于显示数值，cmap 设置颜色风格
    # Rotate the tick labels for better visibility
    plt.xticks(rotation=45)
    plt.title(size_to_title[size])
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_{size}_heatmap.png")
    plt.clf()

baseline_to_label = {
    "teacher": "Teacher",
    "1.3B": "No KD Large",
    "0.3B": "No KD Medium",
    "0.1B": "No KD Base",
}
baseline_results = {
    "teacher": {},
    "1.3B": {},
    "0.3B": {},
    "0.1B": {},
}
# remove "mope" from mia_nams
mia_nams.remove("mope")
for mia in mia_nams:
    baseline_results["teacher"][mia] = load_auc_results(f"results_6k/{model_name}/sft/blackbox_mia/results_{mia}.json")
for mia in mia_nams:
    for size in model_sizes:
        baseline_results[size][mia] = load_auc_results(f"results_6k/{model_name}/student_ref_sft/{size}/blackbox_mia/results_{mia}.json")

baseline_results["teacher"]["mope"] = load_auc_results(f"results_6k/{model_name}/sft/mope/sigma_0.005_models_10/mope_results_loss.json")
for size in model_sizes:
    baseline_results[size]["mope"] = load_auc_results(f"results_6k/{model_name}/student_ref_sft/{size}/mope/sigma_0.005_models_10/mope_results_loss.json")
mia_nams.append("mope")

df = pd.DataFrame(baseline_results).T
plt.figure(figsize=(6,4))
sns.heatmap(
    df, annot=True, cmap=sns.cubehelix_palette(as_cmap=True),
    xticklabels=[mia_to_label[mia] for mia in mia_nams],
    yticklabels=[baseline_to_label[baseline] for baseline in ["teacher", "1.3B", "0.3B", "0.1B"]],
)  # annot=True 用于显示数值，cmap 设置颜色风格
# Rotate the tick labels for better visibility
plt.xticks(rotation=45)
plt.title("OPT Baselines")
plt.tight_layout()
plt.savefig(f"{save_dir}/{model_name}_baseline_heatmap.png")
plt.clf()
