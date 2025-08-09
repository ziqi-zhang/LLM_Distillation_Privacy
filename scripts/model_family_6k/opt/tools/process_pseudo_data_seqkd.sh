BASE_PATH="."

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=$1

GEN_DATA_DIR=$2
PROCESS_DATA_SAVE_DIR=$3
TOKENIZER_MODEL_PATH=$4
MODEL_TYPE=$5


# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
#     --data-dir ${BASE_PATH}/results/gpt2/gen/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/dolly/pseudo \
#     --model-path gpt2-large \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num -1 \
#     --model-type gpt2

# cp ${BASE_PATH}/processed_data/dolly/full/gpt2/valid_0.bin ${BASE_PATH}/processed_data/dolly/pseudo/gpt2/
# cp ${BASE_PATH}/processed_data/dolly/full/gpt2/valid_0.idx ${BASE_PATH}/processed_data/dolly/pseudo/gpt2/
# cp ${BASE_PATH}/processed_data/dolly/full/gpt2/valid.jsonl ${BASE_PATH}/processed_data/dolly/pseudo/gpt2/



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