BASE_PATH="."

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=$1

GEN_DATA_DIR=$2
PROCESS_DATA_SAVE_DIR=$3
TOKENIZER_MODEL_PATH=$4
MODEL_TYPE=$5

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/results/openllama2/gen/openllama2-7B/t1.0-l512 \
#     --processed-data-dir ${BASE_PATH}/processed_data/dolly/pseudo \
#     --model-path ${BASE_PATH}/checkpoints/openllama2-7B \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num -1 \
#     --model-type openllama2

# cp ${BASE_PATH}/processed_data_mia6k/dolly/full/openllama2/valid_0.bin ${BASE_PATH}/processed_data_mia6k/dolly/pseudo/openllama2/
# cp ${BASE_PATH}/processed_data_mia6k/dolly/full/openllama2/valid_0.idx ${BASE_PATH}/processed_data_mia6k/dolly/pseudo/openllama2/
# cp ${BASE_PATH}/processed_data_mia6k/dolly/full/openllama2/valid.jsonl ${BASE_PATH}/processed_data_mia6k/dolly/pseudo/openllama2/

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir $GEN_DATA_DIR \
    --processed-data-dir $PROCESS_DATA_SAVE_DIR \
    --model-path $TOKENIZER_MODEL_PATH \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type $MODEL_TYPE

cp ${BASE_PATH}/processed_data_mia6k/dolly/full/${MODEL_TYPE}/valid_0.bin ${PROCESS_DATA_SAVE_DIR}/${MODEL_TYPE}/
cp ${BASE_PATH}/processed_data_mia6k/dolly/full/${MODEL_TYPE}/valid_0.idx ${PROCESS_DATA_SAVE_DIR}/${MODEL_TYPE}/
cp ${BASE_PATH}/processed_data_mia6k/dolly/full/${MODEL_TYPE}/valid.jsonl ${PROCESS_DATA_SAVE_DIR}/${MODEL_TYPE}/