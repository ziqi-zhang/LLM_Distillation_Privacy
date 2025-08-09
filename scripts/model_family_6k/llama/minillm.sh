#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=9374
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=$(echo $1 | awk -F',' '{print NF}')
BATCH_SIZE=$2
SAVE_PATH=$3
CKPT_NAME=$4
CKPT_PATH=$5
TEACHER_CKPT_NAME=$6
TEACHER_CKPT_PATH=$7
TEACHER_PEFT_CKPT_NAME=$8
TEACHER_PEFT_CKPT_PATH=$9
TEACHER_CKPT=$TEACHER_CKPT_PATH
CKPT=${CKPT_PATH}
TEACHER_PEFT_CKPT=${TEACHER_PEFT_CKPT_PATH}

# 函数：检查端口是否被占用
is_port_in_use() {
    local port=$1
    # 检查端口是否被占用（使用netstat或ss命令）
    if netstat -tuln | grep -q ":$port "; then
        return 0 # 端口被占用
    else
        return 1 # 端口未被占用
    fi
}
# 检查初始端口及后续端口，直到找到一个未被占用的端口
while is_port_in_use $MASTER_PORT; do
    echo "端口 $MASTER_PORT 已被占用，尝试下一个端口..."
    MASTER_PORT=$((MASTER_PORT + 1))
done

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH="."
# CKPT_NAME="openllama2-3B"
# CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# PEFT_CKPT_NAME="openllama2-3B"
# PEFT_CKPT="${BASE_PATH}/results/openllama2/train/init/${PEFT_CKPT_NAME}/"
# TEACHER_CKPT_NAME="openllama2-7B"
# TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
# TEACHER_PEFT_CKPT_NAME="sft_7B"
# TEACHER_PEFT_CKPT="${BASE_PATH}/results/openllama2/train/sft/${TEACHER_PEFT_CKPT_NAME}/"
# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data_mia6k/dolly/prompt/openllama2/"
LM_DATA_DIR="${BASE_PATH}/processed_data_mia6k/openwebtext/openllama2/512/1M/"
# runtime
# SAVE_PATH="${BASE_PATH}/results/openllama2/train/minillm/3B_7B"
# hp
GRAD_ACC=4
BATCH_SIZE=2
CHUNK_SIZE=8


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --epochs 10"
OPTS+=" --total-iters 5000"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
OPTS+=" --scheduler-name cosine_trm"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 500"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# lora
OPTS+=" --peft lora"
OPTS+=" --do-train"
# OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
# OPTS+=" --peft-path ${PEFT_CKPT}"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
OPTS+=" --num-rollouts 256"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"

OPTS+=" --train-data-mode public"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export CUDA_VISIBLE_DEVICES=$1
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
