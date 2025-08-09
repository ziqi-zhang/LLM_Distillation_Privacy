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
TEACHER_CKPT=$TEACHER_CKPT_PATH


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
# CKPT_NAME="1.3B-init"
# CKPT="${BASE_PATH}/results/opt/train/init/opt-1.3B"
CKPT=${CKPT_PATH}
# TEACHER_CKPT_NAME="2.7B-sft"
# TEACHER_CKPT="${BASE_PATH}/results/opt/train/sft/opt-2.7B/"
# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data_mia6k/dolly/prompt/opt/"
LM_DATA_DIR="${BASE_PATH}/processed_data_mia6k/openwebtext/opt/512/1M/"
# runtime
# SAVE_PATH="${BASE_PATH}/results/opt/train/minillm/1.3B_2.7B"
# hp
# GRAD_ACC=2
GRAD_ACC=1
BATCH_SIZE=4
CHUNK_SIZE=8


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --model-type opt"
OPTS+=" --teacher-model-fp16"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num -1"
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
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 5000"
OPTS+=" --eval-interval 5000"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
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
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
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
