


### set up env vairables ###
source ../SETUP_DOCKER_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8

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



# single task finetuning

bash run_xttn_with_multiple_knowlege_augmentation_offline_.sh \
"openbookqa_main" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
2 \
15 \
"t5-base" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_10aug_latent_64.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_8_2_2/8_2_openbook_qa_10aug_latent_64" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
-1 \
1 # gradient_accumulation_step

echo "done 10aug latent 64"


bash run_xttn_with_multiple_knowlege_augmentation_offline_.sh \
"rotten_tomatoes" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
2 \
15 \
"t5-base" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_1aug.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_8_2_2/8_2_rotten_tomatoes_1aug_latent_64" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
-1 \
1 # gradient_accumulation_step

echo "done 1aug latent 64"


bash run_xttn_with_multiple_knowlege_augmentation_offline_.sh \
"rotten_tomatoes" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
2 \
15 \
"t5-base" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_5aug.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_8_2_2/8_2_rotten_tomatoes_5aug_latent_64" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
-1 \
1 # gradient_accumulation_step

echo "done 5aug latent 64"


bash run_xttn_with_multiple_knowlege_augmentation_offline_.sh \
"rotten_tomatoes" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
2 \
15 \
"t5-base" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_10aug_latent_64.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_8_2_2/8_2_rotten_tomatoes_10aug_latent_64" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
-1 \
1 # gradient_accumulation_step

echo "done 10aug latent 64"


