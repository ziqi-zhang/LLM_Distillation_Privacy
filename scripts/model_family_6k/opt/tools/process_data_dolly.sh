BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data_mia6k/dolly/prompt \
    --model-path facebook/opt-1.3B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 4000 \
    --only-prompt \
    --model-type opt

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data_mia6k/dolly/full \
    --model-path facebook/opt-1.3B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 4000 \
    --model-type opt
