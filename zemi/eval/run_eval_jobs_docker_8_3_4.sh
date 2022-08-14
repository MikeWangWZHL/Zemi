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


source ../SETUP_DOCKER_ENV.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8

### debug
# CUDA_VISIBLE_DEVICES=4,5,6,7
# MAIN_PROCESS_PORT=20655
# NUM_PROCESSES=4


# result dir containing checkpoints
RESULT_DIR_ROOT="${OUTPUT_SRC_ROOT}/p3_finetuning"
echo "checkpoint result dir: ${RESULT_DIR_ROOT}"


MODEL_PATH_NAME="run_jobs_docker_8_2_1/7_30_multitask_mixture_v1plus_n_2_c4_5percent_baseline_concat_5_aug"
for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "hellaswag"
do
    bash run_eval_finetuned_mixture_baseline.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${OUTPUT_SRC_ROOT}/eval/run_eval_jobs_docker_8_3_4/${MODEL_PATH_NAME}/${TASK_NAME}" \
    "${RESULT_DIR_ROOT}/${MODEL_PATH_NAME}" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    16 # eval batch size
done