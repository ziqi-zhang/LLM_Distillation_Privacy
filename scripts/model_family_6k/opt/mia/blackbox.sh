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

# for MODEL_SIZE in 1.3B; do
for MODEL_SIZE in "1.3B" "0.3B" "0.1B"; do

# for KD_MODE in kd init seqkd distillm gkd imitkd minillm; do
# for KD_MODE in kd seqkd imitkd gkd distillm minillm; do
for KD_MODE in minillm; do
    BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[${MODEL_SIZE}]}
    FORMAL_NAME=${MODEL_TO_FORMAL_NAME[$MODEL_SIZE]}

    CKPT_NAME="opt-${MODEL_SIZE}"
    CKPT_PATH="facebook/${FORMAL_NAME}"
    KD_SAVE_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/${KD_MODE}"
    CKPT_PATH="${KD_SAVE_PATH}/final"
    MIA_SAVE_PATH="${KD_SAVE_PATH}/blackbox_mia"
    REF_SAME_DOMAIN_CKPT="${ROOT_SAVE_DIR}/student_calibration/final"

    KD_CKPT_FILE_PATH=${CKPT_PATH}/pytorch_model.bin
    if check_file $KD_CKPT_FILE_PATH; then
        echo "文件存在: $SFT_FINAL_CKPT_PATH"
        bash ${SCRIPT_DIR}/eval_main_dolly.sh $GPUS $BATCH_SIZE $MIA_SAVE_PATH $CKPT_NAME $CKPT_PATH
    else
        echo "文件不存在: $SFT_FINAL_CKPT_PATH"
    fi

    
done


# BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[${MODEL_SIZE}]}
# CKPT_NAME="opt-${MODEL_SIZE}"
# KD_SAVE_PATH="${ROOT_SAVE_DIR}/student_ref_sft/${MODEL_SIZE}/"
# CKPT_PATH="${KD_SAVE_PATH}/final"
# MIA_SAVE_PATH="${KD_SAVE_PATH}/blackbox_mia"
# KD_CKPT_FILE_PATH=${CKPT_PATH}/pytorch_model.bin
# if check_file $KD_CKPT_FILE_PATH; then
#     echo "文件存在: $SFT_FINAL_CKPT_PATH"
#     bash ${SCRIPT_DIR}/eval_main_dolly.sh $GPUS $BATCH_SIZE $MIA_SAVE_PATH $CKPT_NAME $CKPT_PATH
# else
#     echo "文件不存在: $SFT_FINAL_CKPT_PATH"
# fi


done