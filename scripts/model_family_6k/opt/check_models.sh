# 定义一个函数来检查文件的属性
check_file() {
    local file_path="$1"

    if [ -e "$file_path" ]; then
        # echo "文件存在: $file_path"
        return 0 # True
    else
        # echo "文件不存在: $file_path"
        return 1 # False
    fi
}

for MODEL_SIZE in "1.3B" "0.3B" "0.1B"; do

for KD in kd init seqkd distillm gkd imitkd minillm; do

KD_DIR=results_6k/opt/${MODEL_SIZE}/${KD}
CKPT_PATH=${KD_DIR}/final/pytorch_model.bin
    if check_file $CKPT_PATH; then
        echo $MODEL_SIZE $KD 文件存在
    else
        echo $MODEL_SIZE $KD 文件不存在
    fi

done
    
done