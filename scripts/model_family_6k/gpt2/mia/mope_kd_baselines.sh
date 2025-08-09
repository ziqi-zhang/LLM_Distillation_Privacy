GPUS=$1
SCRIPT_DIR=$(dirname $0)
ROOT_SAVE_DIR="results_6k/gpt2"

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
    ["large"]=8
    ["medium"]=16
    ["base"]=32
)

declare -A MODEL_TO_HF_NAME=(
    ["large"]="gpt2-large"
    ["medium"]="gpt2-medium"
    ["base"]="gpt2"
)

MODELS=10
SIGMA=0.005
DEV_NUM=-1

# for MODEL_SIZE in large; do
for MODEL_SIZE in medium base; do
# for MODEL_SIZE in large medium base; do

# # for KD_MODE in kd init seqkd distillm gkd imitkd minillm; do
# for KD_MODE in kd seqkd imitkd gkd distillm minillm; do
# # for KD_MODE in kd; do
#     BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[${MODEL_SIZE}]}
#     CKPT_NAME="gpt2-${MODEL_SIZE}"
#     KD_SAVE_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/${KD_MODE}"
#     CKPT_PATH="${KD_SAVE_PATH}"
#     MIA_SAVE_PATH="${KD_SAVE_PATH}/mope"
#     REF_SAME_DOMAIN_CKPT="results/gpt2/student_calibration/final"

#     KD_CKPT_FILE_PATH=${CKPT_PATH}/final/pytorch_model.bin
#     if check_file $KD_CKPT_FILE_PATH; then
#         echo "文件存在: $SFT_FINAL_CKPT_PATH"
#         bash ${SCRIPT_DIR}/mope.sh $GPUS $CKPT_PATH $MODELS $SIGMA $BATCH_SIZE $DEV_NUM

#     else
#         echo "文件不存在: $SFT_FINAL_CKPT_PATH"
#     fi

    
# done

BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[${MODEL_SIZE}]}
CKPT_NAME="gpt2-${MODEL_SIZE}"
KD_SAVE_PATH="${ROOT_SAVE_DIR}/student_ref_sft/${MODEL_SIZE}/"
CKPT_PATH="${KD_SAVE_PATH}"
MIA_SAVE_PATH="${KD_SAVE_PATH}/mope"
KD_CKPT_FILE_PATH=${CKPT_PATH}/final/pytorch_model.bin
if check_file $KD_CKPT_FILE_PATH; then
    echo "文件存在: $SFT_FINAL_CKPT_PATH"
    bash ${SCRIPT_DIR}/mope.sh $GPUS $CKPT_PATH $MODELS $SIGMA $BATCH_SIZE $DEV_NUM

else
    echo "文件不存在: $SFT_FINAL_CKPT_PATH"
fi


done