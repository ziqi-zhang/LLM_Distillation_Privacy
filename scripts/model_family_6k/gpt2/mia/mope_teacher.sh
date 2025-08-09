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
DEV_NUM=1000

for SIGMA in 0.005 0.001 0.01; do

BATCH_SIZE=4
CKPT_NAME="gpt2-xl"
TEACHER_SAVE_PATH="${ROOT_SAVE_DIR}/sft/"
CKPT_PATH="${TEACHER_SAVE_PATH}"
MIA_SAVE_PATH="${TEACHER_SAVE_PATH}/blackbox_mia"
REF_SAME_DOMAIN_CKPT="results_6k/gpt2/student_calibration/"

TEACHER_CKPT_FILE_PATH=${CKPT_PATH}/final/pytorch_model.bin
if check_file $TEACHER_CKPT_FILE_PATH; then
    echo "文件存在: $TEACHER_CKPT_FILE_PATH"
    bash ${SCRIPT_DIR}/mope.sh $GPUS $CKPT_PATH $MODELS $SIGMA $BATCH_SIZE $DEV_NUM
else
    echo "文件不存在: $TEACHER_CKPT_FILE_PATH"
fi

done