GPUS=$1
SCRIPT_DIR=$(dirname $0)
ROOT_SAVE_DIR="results_6k/opt"

check_file() {
    local file_path="$1"

    if [ -e "$file_path" ]; then
        echo "文件存在: $file_path"
        return 0 # True
    else
        echo "文件不存在: $file_path"
        return 1 # False
    fi
}



declare -A MODEL_TRAIN_BATCH_SIZE=(
    ["1.3B"]=4
    ["0.3B"]=8
    ["0.1B"]=16
)

declare -A MODEL_TO_FORMAL_NAME=(
    ["1.3B"]="opt-1.3b"
    ["0.3B"]="opt-350m"
    ["0.1B"]="opt-125m"
)


# Teacher
# BATCH_SIZE=4
# CKPT_NAME="opt-${MODEL_SIZE}"
# KD_SAVE_PATH="${ROOT_SAVE_DIR}/sft"
# CKPT_PATH="${KD_SAVE_PATH}/final"
# MEM_SAVE_PATH="${KD_SAVE_PATH}/memorization"
# KD_CKPT_FILE_PATH=${CKPT_PATH}/pytorch_model.bin
# MEM_SAVE_RESULTS_PATH="${MEM_SAVE_PATH}/memorization_acc.json"
# if check_file $MEM_SAVE_RESULTS_PATH; then
#     echo "文件存在: $MEM_SAVE_RESULTS_PATH"
# else
#     echo "文件不存在: $MEM_SAVE_RESULTS_PATH"
#     bash ${SCRIPT_DIR}/memorize.sh $GPUS $BATCH_SIZE $MEM_SAVE_PATH $CKPT_NAME $CKPT_PATH
# fi



# for MODEL_SIZE in "1.3B" "0.3B" "0.1B"; do
for MODEL_SIZE in "1.3B"; do
# for MODEL_SIZE in "0.3B" "0.1B"; do


BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[${MODEL_SIZE}]}
CKPT_NAME="opt-${MODEL_SIZE}"
KD_SAVE_PATH="${ROOT_SAVE_DIR}/student_ref_sft/${MODEL_SIZE}/"
CKPT_PATH="${KD_SAVE_PATH}/final"
MEM_SAVE_PATH="${KD_SAVE_PATH}/memorization"
KD_CKPT_FILE_PATH=${CKPT_PATH}/pytorch_model.bin
MEM_SAVE_RESULTS_PATH="${MEM_SAVE_PATH}/memorization_acc.json"
if check_file $MEM_SAVE_RESULTS_PATH; then
    echo "文件存在: $MEM_SAVE_RESULTS_PATH"
else
    echo "文件不存在: $MEM_SAVE_RESULTS_PATH"
    bash ${SCRIPT_DIR}/memorize.sh $GPUS $BATCH_SIZE $MEM_SAVE_PATH $CKPT_NAME $CKPT_PATH
fi

for KD_MODE in kd seqkd imitkd gkd distillm minillm; do
# for KD_MODE in kd; do

    BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[${MODEL_SIZE}]}
    CKPT_NAME="opt-${MODEL_SIZE}"
    KD_SAVE_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/${KD_MODE}"
    CKPT_PATH="${KD_SAVE_PATH}/final"
    MEM_SAVE_PATH="${KD_SAVE_PATH}/memorization"
    KD_CKPT_FILE_PATH=${CKPT_PATH}/pytorch_model.bin
    MEM_SAVE_RESULTS_PATH="${MEM_SAVE_PATH}/memorization_acc.json"
    # if check_file $MEM_SAVE_RESULTS_PATH; then
    #     echo "文件存在: $MEM_SAVE_RESULTS_PATH"
    # else
    #     echo "文件不存在: $MEM_SAVE_RESULTS_PATH"
    #     bash ${SCRIPT_DIR}/memorize.sh $GPUS $BATCH_SIZE $MEM_SAVE_PATH $CKPT_NAME $CKPT_PATH
    # fi
    bash ${SCRIPT_DIR}/memorize.sh $GPUS $BATCH_SIZE $MEM_SAVE_PATH $CKPT_NAME $CKPT_PATH

done

done