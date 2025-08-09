import os, sys
import os.path as osp
import json
import numpy as np
import pandas as pd

model_name = "gpt2"
kd_names = ["kd", "seqkd", "gkd", "imitkd", "minillm", "distillm"]
mia_nams = ["loss", "mink", "minkpp", "ref", "ref_pretrain_teacher", "zlib"]
model_sizes = ["large", "medium", "base"]

def load_auc_results(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results["AUC"]

            
results = {}
for mia in mia_nams:
    results[mia] = {}
    for size in model_sizes:
        results[mia][size] = {}
        for kd in kd_names:
            results[mia][size][kd] = load_auc_results(f"results/{model_name}/{size}/{kd}/blackbox_mia/results_{mia}.json")
            
results["mope"] = {}
for size in model_sizes:
    results["mope"][size] = {}
    for kd in kd_names:
        results["mope"][size][kd] = load_auc_results(f"results/{model_name}/{size}/{kd}/mope/sigma_0.01_models_10/mope_results_loss.json")

mia_nams.append("mope")
            
# Average to each model size
model_size_results = {}
for size in model_sizes:
    per_size_results = []
    for kd in kd_names:
        for mia in mia_nams:
            per_size_results.append(results[mia][size][kd])
    model_size_results[size] = np.mean(per_size_results)
print(model_size_results)
df = pd.DataFrame(model_size_results, index=["AUC"])
print(df)

# Average to each MIA
mia_results = {}
for mia in mia_nams:
    per_mia_results = []
    for size in model_sizes:
        for kd in kd_names:
            per_mia_results.append(results[mia][size][kd])
    mia_results[mia] = np.mean(per_mia_results)
df = pd.DataFrame(mia_results, index=["AUC"])
print(df)

# Average to each KD
kd_results = {}
for kd in kd_names:
    per_kd_results = []
    for size in model_sizes:
        for mia in mia_nams:
            per_kd_results.append(results[mia][size][kd])
    kd_results[kd] = np.mean(per_kd_results)
df = pd.DataFrame(kd_results, index=["AUC"])
print(df)


flat_data = []
for outer_key, inner_dict in results.items():
    flattened_row = {}
    for mid_key, values_dict in inner_dict.items():
        for inner_key, value in values_dict.items():
            flattened_row[f'{mid_key}_{inner_key}'] = value
    flattened_row['Category'] = outer_key  # 添加一个标识顶层字典的列
    flat_data.append(flattened_row)
        
df = pd.DataFrame(flat_data)
df.set_index('Category', inplace=True)
print(df)
    
