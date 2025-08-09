GPUS=$1
SCRIPT_DIR=$(dirname $0)
ROOT_SAVE_DIR="results_6k/llama"

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



BATCH_SIZE=16
STUDENT_PRETRAIN_CKPT_NAME="openllama2-3B"
STUDENT_PRETRAIN_CKPT_PATH="openlm-research/open_llama_3b"

for KD_MODE in kd seqkd imitkd gkd distillm minillm; do
# for KD_MODE in kd; do

    KD_SAVE_PATH="${ROOT_SAVE_DIR}/${MODEL_SIZE}/${KD_MODE}"
    STUDENT_PEFT_CKPT_NAME="${KD_MODE}_3B"
    STUDENT_PEFT_CKPT="${KD_SAVE_PATH}/final"
    MIA_SAVE_PATH="${KD_SAVE_PATH}/blackbox_mia"
    REF_SAME_DOMAIN_CKPT="${ROOT_SAVE_DIR}/student_calibration/final"

    KD_CKPT_FILE_PATH=${STUDENT_PEFT_CKPT}/adapter_model.bin
    if check_file $KD_CKPT_FILE_PATH; then
        echo "文件存在: $SFT_FINAL_CKPT_PATH"
        bash ${SCRIPT_DIR}/eval_main_dolly.sh \
        $GPUS $BATCH_SIZE $MIA_SAVE_PATH \
        $STUDENT_PRETRAIN_CKPT_NAME $STUDENT_PRETRAIN_CKPT_PATH $STUDENT_PEFT_CKPT_NAME $STUDENT_PEFT_CKPT
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

