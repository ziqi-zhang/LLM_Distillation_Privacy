
# Membership and Memorization in LLM Knowledge Distillation

<!-- <a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a> -->

Pytorch implementation for the paper: \
**Membership and Memorization in LLM Knowledge Distillation** \
Ziqi Zhang, Ali Shahin Shamsabadi, Hanxiao Lu, Yifeng Cai, Hamed Haddadi


## Scripts

The scripts for membership inference on distilled models are in `scripts/model_family_6k`. The scripts of memorization is in `mia`. 

## Environment
```bash
bash install.sh
```

Our code is based on [this commit](https://github.com/huggingface/transformers/commit/85fde09c97213bf7e8625f83096bb2a9e183f987) of HuggingFace Transformers **by following DistiLLM and MiniLLM**.

## Data
### Resources
+ The training/evaluation intruction-response data before processing can be downloaded from this [link](https://conversationhub.blob.core.windows.net/beit-share-public/MiniLLM/data.tar?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D).
+ The plain-text corpus $\mathcal{D}_\text{PT}$ can be download from the HugginFace datasets [repository](https://huggingface.co/datasets/openwebtext).


### Data Processing
Get plain-text corpus $\mathcal{D}_\text{PT}$:
```bash
python3 tools/get_openwebtext.py
```
This script will replace the continuous `\n` in each document with a special token "<@x(x!>" and write each document in OpenWebText in a line, which is convenient for parallel processing. In `data/openwebtext/data.txt`, we give an example of the resulting format. You can follow this format to prepare other corpus beyond OpenWebText.

Tokenize the data and store them in binary files:
```bash
bash scripts/gpt2/tools/process_data_dolly.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_pretrain.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process OpenWebText Train / Validation Data

bash scripts/opt/tools/process_data_dolly.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process Dolly Train / Validation Data
bash scripts/opt/tools/process_data_pretrain.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process OpenWebText Corpus Train / Validation Data

bash scripts/llama/tools/process_data_dolly.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process Dolly Train / Validation Data
bash scripts/llama/tools/process_data_pretrain.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process OpenWebText Corpus Train / Validation Data
```

## Base Pre-trained Models
To run fine-tuning or standard KD baselines, you need to download the model checkpoints from [Huggingface Model Hub] and put them in `checkpoints/`. For example, for gpt2-large, you can download the model from this [link](https://huggingface.co/gpt2-large/tree/main) and put them in `checkpoints/gpt2-large`.

Alternatively, you can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download the base models automatically. For example, set `CKPT="gpt2-large"` in `scripts/gpt2/sft/sft_large.sh` causes download of the gpt2-large base model from the HugginFace model hub.

## Train
We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/opt` and `scripts/openllama2`. All our experiments are conducted on 4 \* 40A100, which can be reduced for small models.

### Baselines
The final checkpoints are selected by the **ROUGE-L** scores.

#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```
#### SFT Baselines
```bash
bash scripts/gpt2/sft/sft_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/sft/sft_medium.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/sft/sft_large.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### KD Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/kd/kd_medium.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/kd/kd_large.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### SeqKD Baselines
Generate and process responses with the teacher:
```bash
bash scripts/gpt2/tools/generate_data_seqkd.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/tools/process_pseudo_data_seqkd.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```
Fine-tune the model with SeqKD:
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/seqkd/seqkd_medium.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/seqkd/seqkd_large.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### Student Initialization
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/init/init_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/init/init_medium.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/init/init_large.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### ImitKD Baselines
```bash
bash scripts/gpt2/imitkd/imitkd_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/imitkd/imitkd_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/imitkd/imitkd_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### MiniLLM Baselines
```bash
bash scripts/gpt2/minillm/train_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/minillm/train_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/minillm/train_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### GKD Baselines
```bash
bash scripts/gpt2/gkd/gkd_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/gkd/gkd_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/gkd/gkd_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### DistiLLM
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/init/init_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/init/init_medium.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/init/init_large.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

The final checkpoints are selected by the **ROUGE-L** scores.
```bash
bash scripts/gpt2/distillm/train_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/distillm/train_medium_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/distillm/train_large_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

## Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/DistiLLM}
bash scripts/opt/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/DistiLLM} 
bash scripts/openllama2/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/DistiLLM} 
```

## Results
DistiLLM outperforms other KD baselines in terms of both generation performance and training speed for various model families such as GPT-2, OPT, and OpenLLaMA.
<p align="center">
<img width="1394" src="https://github.com/jongwooko/distillm/assets/59277369/19ddac5c-4cd6-4d81-99d8-32723a8e60d8">
</p>

## Checkpoints (OpenLLaMA-3B)
We share the LoRA weights for OpenLLaMA-3B in [google drive](https://drive.google.com/drive/folders/1Yun1aNpn-mz2h-IVH_VdJ1Jhzm0K55Bo?usp=sharing).

## Acknowledgement
The code framework is based on [DistiLLM](https://github.com/jongwooko/distillm) and [MiniLLM](https://github.com/kuleshov/minillm)

