### set up env vairables ###
source SETUP_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8
MIXTURE="cos_e_v1.11 cosmos_qa dream qasc quartz sciq social_i_qa wiqa"

#############################
## training arg formats
    # DATASET_DIR_NAMES=$1               # training mixture
    # DATASET_ROOT=$2                    # dir containing the huggingface datasets
    # LR=$3                              # learning rate
    # BS=$4                              # training batch size
    # EP=$5                              # training epoch
    # LM_NAME=$6                         # backbone language model name
    # PERCEIVER_CONFIG=$7                # perceiver resampler config json path
    # OUTPUT_DIR=$8                      # output dir 
    # WANDB_RUN_NAME=$9                  # (default disabled) for inspecting training progress 
    # CUDA_VISIBLE_DEVICES=${10}         # CUDA devices
    # --main_process_port ${11}          # for accelerate config 
    # --num_processes ${12}              # for accelerate config
    # model architecture ${13}           # model architecture name
    # N ${14}                            # how many templates to use for training, -1 for using all, default = 2
    # --gradient_accumulation_step ${15} # default = 1
#############################

bash ./training/run_xttn_with_multiple_knowlege_augmentation_offline_FiD_.sh \
"${MIXTURE}" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
1 \
10 \
"google/t5-large-lm-adapt" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_5aug_t5_large.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/fid_large" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"FiD" \
2 \
1

echo "done training"


### input format ###
    # DATASET_DIR_NAME=$1       # dataset names to be evaluated on  
    # DATASETS_ROOT=$2          # dir containing the huggingface datasets
    # MODEL_PATH=$3             # trained model checkpoint dir
    # OUTPUT_DIR=$4             # output dir
    # CUDA_VISIBLE_DEVICES=$5   # CUDA devices
    # --main_process_port $6    # for accelerate config 
    # --num_processes $7        # for accelerate config
    # eval_batch $8             # eval batch size
##############################

for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "hellaswag"
do
    bash ./eval/run_eval_finetuned_mixture_xattn_FiD.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${OUTPUT_SRC_ROOT}/p3_finetuning/fid_large" \
    "${OUTPUT_SRC_ROOT}/eval/fid_large/${TASK_NAME}" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    1 # eval batch size
done