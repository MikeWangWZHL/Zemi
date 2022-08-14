


### set up env vairables ###
source SETUP_DOCKER_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8

## debug
# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4

### test xattn ###
#############################
## arg_formats (multi task)
    # DATASET_DIR_NAMES=$1
    # DATASET_ROOT=$2
    # LR=$3
    # BS=$4
    # EP=$5
    # LM_NAME=$6
    # PERCEIVER_CONFIG=$7
    # OUTPUT_DIR=$8
    # WANDB_RUN_NAME=$9
    # CUDA_VISIBLE_DEVICES=${10}
    # --main_process_port ${11} 
    # --num_processes ${12}
    # model architecture ${13}
    # N ${14} how many templates to use for training, -1 for using all
    # --gradient_accumulation_step # default = 1
#############################

#### single task #####
bash ./training/run_xttn_with_multiple_knowlege_augmentation_offline_.sh \
"openbookqa_main" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug_cross_shuffled" \
0.0001 \
2 \
15 \
"t5-base" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_5aug.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_train_eval_8_4_9__cross_shuffled_single_task/8_4_openbooqa_5aug_cross_shuffled" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
-1 \
1 # gradient_accumulation_step

echo "done learning rate"


### input format ###
    # DATASET_DIR_NAME=$1
    # DATASETS_ROOT=$2 # -> DATASETS_DIR="${DATASETS_ROOT}/${DATASET_DIR_NAME}/*"
    # MODEL_PATH=$3
    # OUTPUT_DIR=$4
    # CUDA_VISIBLE_DEVICES=$5
    # --main_process_port $6
    # --num_processes $7
    # eval_batch $8
##############################

for TASK_NAME in "openbookqa_main"
do
    bash ./eval/run_eval_finetuned_mixture_xattn.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug_cross_shuffled" \
    "${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_train_eval_8_4_9__cross_shuffled_single_task/8_4_openbooqa_5aug_cross_shuffled" \
    "${OUTPUT_SRC_ROOT}/eval/run_jobs_docker_train_eval_8_4_9__cross_shuffled_single_task/8_4_openbooqa_5aug_cross_shuffled/${TASK_NAME}" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    8 # eval batch size
done