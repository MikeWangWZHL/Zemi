


### set up env vairables ###
source ../SETUP_DOCKER_ENV.sh

### orignal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8
MIXTURE="cos_e_v1.11 cosmos_qa dream qasc quartz sciq social_i_qa wiqa wiki_qa trec super_glue_copa rotten_tomatoes"

### local test
# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4
# MIXTURE="super_glue_copa rotten_tomatoes"

####################################
    # DATASET_DIR_NAMES=$1
    # DATASET_ROOT=$2
    # MODEL_NAME_OR_PATH=$3 # "google/t5-base-lm-adapt"
    # LR=$4 # 0.0001
    # BS=$5 # 4
    # EP=$6 # 10
    # N=$7 # 2
    # OUTPUT_DIR=$8
    # WANDB_RUN_NAME=$9
    # CONCAT_AUG_NUM=${10}
    # CUDA_VISIBLE_DEVICES=${11}
    # --main_process_port ${12} 
    # --num_processes ${13}
    # --gradient_accumulation_step # default = 1
####################################

echo ${MIXTURE}

bash run_mixture_baseline_offline.sh \
"${MIXTURE}" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
"google/t5-base-lm-adapt" \
0.0001 \
4 \
10 \
2 \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_8_2_0/7_30_multitask_mixture_v1plus_n_2_c4_5percent_baseline_no_aug" \
"not_using_wandb" \
0 \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
1
echo "done multi-task no aug"



