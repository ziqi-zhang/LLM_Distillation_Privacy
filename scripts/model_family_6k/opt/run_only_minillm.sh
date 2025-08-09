GPUS=$1
SCRIPT_DIR=$(dirname $0)
ROOT_SAVE_DIR="results_6k/opt"

# 定义一个函数来检查文件的属性
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


SFT_SAVE_PATH="${ROOT_SAVE_DIR}/sft"
SFT_FINAL_CKPT_DIR="${SFT_SAVE_PATH}"
SFT_FINAL_CKPT_PATH="${SFT_FINAL_CKPT_DIR}/final/pytorch_model.bin"
if check_file $SFT_FINAL_CKPT_PATH; then
    echo "文件存在: $SFT_FINAL_CKPT_PATH"
else
    echo "文件不存在: $SFT_FINAL_CKPT_PATH"
    bash $SCRIPT_DIR/teacher_sft.sh $GPUS $SFT_FINAL_CKPT_DIR
fi

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

# Generate pseudo data for seqKD
GEN_SAVE_PATH=${SFT_FINAL_CKPT_DIR}/gen
TEACHER_CKPT_NAME="2.7B-sft"
TEACHER_CKPT_PATH=$SFT_FINAL_CKPT_DIR/final
GEN_SAVE_PATH_FILE="${GEN_SAVE_PATH}/raw.jsonl"
if check_file $GEN_SAVE_PATH_FILE; then
    echo "文件存在: $GEN_SAVE_PATH_FILE"
else
    echo "文件不存在: $GEN_SAVE_PATH_FILE"
    bash $SCRIPT_DIR/tools/generate_data_seqkd.sh $GPUS $GEN_SAVE_PATH $TEACHER_CKPT_NAME $TEACHER_CKPT_PATH
fi


PROCESS_DATA_SAVE_DIR="./processed_data_mia6k/dolly/pseudo"
TOKENIZER_MODEL_PATH="facebook/opt-2.7B"
MODEL_TYPE="opt"
MODEL_SAVE_NAME="opt-${TEACHER_CKPT_NAME}"
PSEUDO_DATA_DIR="${PROCESS_DATA_SAVE_DIR}/${MODEL_TYPE}"
PROCESS_DATA_SAVE_FILE="${PSEUDO_DATA_DIR}/train_0.bin"
if check_file $PROCESS_DATA_SAVE_FILE; then
    echo "文件存在: $PROCESS_DATA_SAVE_FILE"
else
    echo "文件不存在: $PROCESS_DATA_SAVE_FILE"
    bash $SCRIPT_DIR/tools/process_pseudo_data_seqkd.sh $GPUS $GEN_SAVE_PATH $PROCESS_DATA_SAVE_DIR $TOKENIZER_MODEL_PATH $MODEL_TYPE
fi



TEACHER_CKPT_NAME="opt-2.7B-sft"
TEACHER_CKPT_PATH=$SFT_FINAL_CKPT_DIR/final

for MODEL_SIZE in "1.3B" "0.3B" "0.1B"; do
# for MODEL_SIZE in "1.3B"; do
# for MODEL_SIZE in "0.3B" "0.1B"; do
    BATCH_SIZE=${MODEL_TRAIN_BATCH_SIZE[$MODEL_SIZE]}
    FORMAL_NAME=${MODEL_TO_FORMAL_NAME[$MODEL_SIZE]}

    CKPT_NAME="opt-${MODEL_SIZE}"
    CKPT_PATH="facebook/${FORMAL_NAME}"
    # for KD_MODE in kd init; do
    # for KD_MODE in seqkd; do
    #     echo "KD_MODE: $KD_MODE"

    #     KD_SAVE_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/${KD_MODE}"
    #     KD_SAVE_CKPT_PATH="${KD_SAVE_PATH}/final/pytorch_model.bin"
    #     if check_file $KD_SAVE_CKPT_PATH; then
    #         echo "$KD_MODE 文件存在: $KD_SAVE_CKPT_PATH"
    #         bash $SCRIPT_DIR/$KD_MODE.sh $GPUS $BATCH_SIZE $KD_SAVE_PATH $CKPT_NAME $CKPT_PATH $TEACHER_CKPT_NAME $TEACHER_CKPT_PATH
    #     else
    #         echo "$KD_MODE 文件不存在: $KD_SAVE_CKPT_PATH"
            
    #     fi
    # done

    INIT_CKPT_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/init/final"
    # for KD_MODE in distillm gkd imitkd minillm; do
    for KD_MODE in minillm; do
        echo "KD_MODE: $KD_MODE"
        KD_SAVE_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/${KD_MODE}"
        KD_SAVE_CKPT_PATH="${KD_SAVE_PATH}/final/pytorch_model.bin"
        if check_file $KD_SAVE_CKPT_PATH; then
            echo "$KD_MODE 文件存在: $KD_SAVE_CKPT_PATH"
        else
            echo "$KD_MODE 文件不存在: $KD_SAVE_CKPT_PATH"
            bash $SCRIPT_DIR/$KD_MODE.sh $GPUS $BATCH_SIZE $KD_SAVE_PATH $CKPT_NAME $CKPT_PATH $TEACHER_CKPT_NAME $TEACHER_CKPT_PATH $INIT_CKPT_PATH
        fi
    done


    SFT_SAVE_PATH="${ROOT_SAVE_DIR}/student_ref_sft/${MODEL_SIZE}"
    SFT_FINAL_CKPT_DIR="${SFT_SAVE_PATH}"
    SFT_FINAL_CKPT_PATH="${SFT_FINAL_CKPT_DIR}/final/pytorch_model.bin"
    # if check_file $SFT_FINAL_CKPT_PATH; then
    #     echo "文件存在: $SFT_FINAL_CKPT_PATH"
    # else
    #     echo "文件不存在: $SFT_FINAL_CKPT_PATH"
    #     bash $SCRIPT_DIR/student_ref_sft.sh $GPUS $SFT_FINAL_CKPT_DIR $CKPT_NAME $CKPT_PATH
    # fi


done