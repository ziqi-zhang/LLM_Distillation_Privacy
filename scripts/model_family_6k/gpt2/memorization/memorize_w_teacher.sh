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
CKPT=$CKPT_PATH



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
# CKPT_NAME="gpt2-base"
# CKPT_ROOT=$2
# echo ${CKPT_ROOT}
# CKPT="${CKPT_ROOT}/final"
# echo ${CKPT}
# exit 0
# data
DATA_NAMES="dolly"
DATA_DIR="${BASE_PATH}/processed_data_mia6k/dolly/full/gpt2/"
# hp
EVAL_BATCH_SIZE=16
# runtime
# SAVE_PATH="${BASE_PATH}/results/gpt2/eval_main/"
# SAVE_PATH="${CKPT_ROOT}/blackbox_mia"
TYPE="mia"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type gpt2"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 576"
OPTS+=" --min-length 576"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
# OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type ${TYPE}"
# gen
# OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

OPTS+=" --train-data-mode private"

OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT_PATH}"



export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
export CUDA_VISIBLE_DEVICES=$1
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate_memorization_w_teacher.py ${OPTS} $@"

# export CUDA_VISIBLE_DEVICES=0
# CMD="python ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
