### set up ENV same as in ../SETUP_DOCKER_ENV.sh ###
# export DATA_ROOT="/data"
# export TRAINING_SRC_ROOT="/code/training"
# export EVAL_SRC_ROOT="/code/eval"
# export OUTPUT_SRC_ROOT="/code/output"
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_HOME=/cache/huggingface
###################


STEP_NUM=-1

DATASET_DIR_NAME=$1
DATASETS_ROOT=$2
DATASETS_PATHS="${DATASETS_ROOT}/${DATASET_DIR_NAME}/*"
OUTPUT_DIR=$3
MODEL_PATH=$4

BS_EVAL=$8

echo $DATASETS_PATHS

CUDA_VISIBLE_DEVICES=$5 accelerate launch --main_process_port $6 --num_processes $7 eval_original_task_only.py \
    --processed_dataset_paths ${DATASETS_PATHS} \
    --model_name_or_path ${MODEL_PATH} \
    --step_num ${STEP_NUM} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size ${BS_EVAL}

