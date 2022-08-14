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

# source ../SETUP_DOCKER_ENV.sh
source SETUP_DOCKER_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8

# ### debug
# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4


# result dir containing checkpoints
RESULT_DIR_ROOT="${OUTPUT_SRC_ROOT}/p3_finetuning"
echo "checkpoint result dir: ${RESULT_DIR_ROOT}"


# MODEL_PATH_NAME="7_30_multitask_mixture_mulcqa_n_2_c4_5percent_30aug_latent_64_FrozenAugEncoder"
# for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "wiki_qa" "hellaswag"
# do
#     bash run_eval_finetuned_mixture_xattn.sh \
#     ${TASK_NAME} \
#     "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
#     "${RESULT_DIR_ROOT}/${MODEL_PATH_NAME}" \
#     "${OUTPUT_SRC_ROOT}/eval/run_eval_jobs_docker_8_3_2/${MODEL_PATH_NAME}/${TASK_NAME}" \
#     ${CUDA_VISIBLE_DEVICES} \
#     ${MAIN_PROCESS_PORT} \
#     ${NUM_PROCESSES} \
#     6 # eval batch size
# done

MODEL_PATH_NAME="7_30_multitask_mixture_mulcqa_n_2_c4_5percent_30aug_latent_64_FrozenAugEncoder"
for TASK_NAME in "hellaswag"
do
    # bash run_eval_finetuned_mixture_xattn.sh \
    bash ./eval/run_eval_finetuned_mixture_xattn.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${RESULT_DIR_ROOT}/${MODEL_PATH_NAME}" \
    "${OUTPUT_SRC_ROOT}/eval/run_eval_jobs_docker_8_3_2/${MODEL_PATH_NAME}/${TASK_NAME}" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    4 # eval batch size
done