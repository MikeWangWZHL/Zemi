


### set up env vairables ###
source SETUP_DOCKER_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8
MIXTURE="cos_e_v1.11 cosmos_qa dream qasc quartz sciq social_i_qa wiqa"

## debug
# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4
# MIXTURE="dream qasc"

# MIXTURE="dream"

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
    # AUG_MAX_LENGTH
    # FIT_AUG_NUM # number of augs to fit into the AUG_MAX_LENGTH
#############################

bash ./training/run_xttn_with_multiple_knowlege_augmentation_fit_aug_num_offline_.sh \
"${MIXTURE}" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
0.0001 \
4 \
10 \
"google/t5-base-lm-adapt" \
"${TRAINING_SRC_ROOT}/perceiver_configs/xattn_multi_aug_config_v1_v3_5aug_latent_128.json" \
"${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_train_eval_8_4_2__fit_10_to_5aug_max_512_latent_128/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_fit_10_to_5aug_max_512_latent_128" \
"not_using_wandb" \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
"SharedEncoderDecoder_MultiAug" \
2 \
1 \
512 \
10 # fit 10 aug -> 5 positions

echo "done training"


### input format ###
    # DATASET_DIR_NAME=$1
    # DATASETS_ROOT=$2 # -> DATASETS_DIR="${DATASETS_ROOT}/${DATASET_DIR_NAME}/*"
    # MODEL_PATH=$3
    # OUTPUT_DIR=$4
    # CUDA_VISIBLE_DEVICES=$5
    # --main_process_port $6
    # --num_processes $7
    # eval_batch $8
    # AUG_MAX_LENGTH=$9
    # FIT_AUG_NUM=${10}
##############################

for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "wiki_qa" "hellaswag"
do
    bash ./eval/run_eval_finetuned_mixture_xattn_fit_aug_num.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${OUTPUT_SRC_ROOT}/p3_finetuning/run_jobs_docker_train_eval_8_4_2__fit_10_to_5aug_max_512_latent_128/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_fit_10_to_5aug_max_512_latent_128" \
    "${OUTPUT_SRC_ROOT}/eval/run_jobs_docker_train_eval_8_4_2__fit_10_to_5aug_max_512_latent_128/8_4_multitask_mixture_mulcqa_n_2_c4_5percent_fit_10_to_5aug_max_512_latent_128/${TASK_NAME}" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    4 \
    512 \
    10
done
