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

BATCH_SIZE=4
CKPT_NAME="opt-1.3b"
TEACHER_SAVE_PATH="${ROOT_SAVE_DIR}/sft/"
CKPT_PATH="${TEACHER_SAVE_PATH}/final"
MIA_SAVE_PATH="${TEACHER_SAVE_PATH}/blackbox_mia"
REF_SAME_DOMAIN_CKPT="results/opt/student_calibration/final"

TEACHER_CKPT_FILE_PATH=${CKPT_PATH}/pytorch_model.bin
if check_file $TEACHER_CKPT_FILE_PATH; then
    echo "文件存在: $TEACHER_CKPT_FILE_PATH"
    bash ${SCRIPT_DIR}/eval_main_dolly.sh $GPUS $BATCH_SIZE $MIA_SAVE_PATH $CKPT_NAME $CKPT_PATH
else
    echo "文件不存在: $TEACHER_CKPT_FILE_PATH"
fi

