BASE_PATH=${1}

MAX_LENGTH=512

CUDA_VISIBLE_DEVICES=${2} \
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data_mia6k/openwebtext/openllama2/${MAX_LENGTH}/ \
    --model-path openlm-research/open_llama_3b \
    --max-length ${MAX_LENGTH} \
    --train-num 1000000 \
    --data-process-workers 32 \
    --dev-num 10000 \
