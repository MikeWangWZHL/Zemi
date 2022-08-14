### set up env vairables ###
source SETUP_DOCKER_ENV.sh

### orignal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8
MIXTURE="cos_e_v1.11 cosmos_qa dream qasc quartz sciq social_i_qa wiqa"

### local test
# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4
# MIXTURE="dream qasc"

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

bash ./training/run_mixture_baseline_offline.sh \
"${MIXTURE}" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
"google/t5-large-lm-adapt" \
0.0001 \
1 \
10 \
2 \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_train_eval_8_4_6__t5_large_baseline_no_aug/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_t5_large_baseline_no_aug" \
"not_using_wandb" \
0 \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
1
echo "done t5-large no aug"


#####
    # DATASET_DIR_NAME=$1
    # DATASETS_DIR=$2
    # OUTPUT_DIR=$3
    # MODEL_PATH=$4
    # CUDA_VISIBLE_DEVICES=$5 
    # --main_process_port $6 
    # --num_processes $7
    # eval_bs $8
#####

### eval ###

for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "wiki_qa" "hellaswag"
do
    bash ./eval/run_eval_finetuned_mixture_baseline.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${OUTPUT_SRC_ROOT}/eval/run_jobs_docker_train_eval_8_4_6__t5_large_baseline_no_aug/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_t5_large_baseline_no_aug/${TASK_NAME}" \
    "${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_train_eval_8_4_6__t5_large_baseline_no_aug/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_t5_large_baseline_no_aug" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    2 # eval batch size
done