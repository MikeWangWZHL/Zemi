################
# delay start for some time
# python sleeping_foo.py --secs 7000
################

DATASET_DIR_NAMES=$1
DATASET_ROOT=$2
MODEL_NAME_OR_PATH=$3 # "google/t5-base-lm-adapt"

MAX_LENGTH=1024
TARGET_MAX_LENGTH=256

LR=$4 # 0.0001
BS=$5 # 4
BS_EVAL=8
EP=$6 # 10
N=$7 # 2


OUTPUT_DIR=$8
WANDB_RUN_NAME=$9

CONCAT_AUG_NUM=${10}

GRADIENT_ACCUMULATION_STEPS=${14}

CUDA_VISIBLE_DEVICES=${11} accelerate launch --main_process_port ${12} --num_processes ${13} multi_task_fine_tune_baseline.py \
    --max_length ${MAX_LENGTH} \
    --target_max_length ${TARGET_MAX_LENGTH} \
    --dataset_dir_names ${DATASET_DIR_NAMES} \
    --dataset_root_path ${DATASET_ROOT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.25 \
    --per_device_train_batch_size ${BS} \
    --per_device_eval_batch_size ${BS_EVAL} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LR} \
    --num_proc 16 \
    --num_train_epochs ${EP} \
    --lr_scheduler_type cosine \
    --save_model \
    --sample_n ${N} \
    --concat_aug_num ${CONCAT_AUG_NUM}
    # --wandb_proj "summer_22_intern" \
    # --wandb_run_name ${WANDB_RUN_NAME} \

