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

source ../SETUP_ENV.sh

echo "DATA_ROOT : ${DATA_ROOT}"
echo "TRAINING_SRC_ROOT : ${TRAINING_SRC_ROOT}"
echo "EVAL_SRC_ROOT : ${EVAL_SRC_ROOT}"
echo "OUTPUT_SRC_ROOT : ${OUTPUT_SRC_ROOT}"
echo "HF_DATASETS_OFFLINE : ${HF_DATASETS_OFFLINE}"
echo "TRANSFORMERS_OFFLINE : ${TRANSFORMERS_OFFLINE}"

#####


# TASK_NAME="openbookqa_main"
MODEL_PATH_NAME="7_27_multitask_mixture_mulcqa_n_2_c4_5percent_concat_baseline/t5-base/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_concat_baseline_5aug"
for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "hellaswag"
do
    bash ./eval/run_eval_finetuned_mixture_baseline.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${OUTPUT_SRC_ROOT}/eval/${MODEL_PATH_NAME}/${TASK_NAME}" \
    "${OUTPUT_SRC_ROOT}/p3_finetuning/${MODEL_PATH_NAME}" \
    4,5,6,7 \
    20655 \
    4
done