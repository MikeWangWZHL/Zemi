


### set up env vairables ###
source ../SETUP_DOCKER_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8

# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4

# MIXTURE="dream"
# MIXTURE="super_glue_copa rotten_tomatoes"
MIXTURE="cos_e_v1.11 cosmos_qa dream qasc quartz sciq social_i_qa wiqa wiki_qa trec super_glue_copa rotten_tomatoes"

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

echo ${MIXTURE}


bash run_xttn_with_multiple_knowlege_augmentation_offline_.sh \
"${MIXTURE}" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
4 \
10 \
"google/t5-base-lm-adapt" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_5aug.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/7_30_multitask_mixture_v1plus_n_2_c4_5percent_5aug_latent_64" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
2 \
1 # gradient_accumulation_step

echo "done 10aug latent 64"


