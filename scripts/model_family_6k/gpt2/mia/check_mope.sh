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

for MODEL_SIZE in large medium base; do

for KD in kd init seqkd distillm gkd imitkd; do

KD_DIR=results/gpt2/${MODEL_SIZE}/${KD}

MOPE_PATH=${KD_DIR}/mope/sigma_0.01_models_10/mope_results_loss.json
if check_file $MOPE_PATH; then
    echo $MODEL_SIZE $KD MoPe 存在
else
    echo $MODEL_SIZE $KD MoPe不存在
fi

BLIMP_PATH=${KD_DIR}/blimp/sigma_0.01_models_10/loss_layer_auc.png
if check_file $BLIMP_PATH; then
    echo $MODEL_SIZE $KD BLiMP 存在
else
    echo $MODEL_SIZE $KD BLiMP不存在
fi

done
    
done