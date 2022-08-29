### set up env vairables ###
source SETUP_ENV.sh

### orignal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=20655
NUM_PROCESSES=8
MIXTURE="cos_e_v1.11 cosmos_qa dream qasc quartz sciq social_i_qa wiqa"

### training arg format ###
####################################
    # DATASET_DIR_NAMES=$1          # training mixture
    # DATASET_ROOT=$2               # dir containing the huggingface datasets                           
    # MODEL_NAME_OR_PATH=$3         # backbone lm model name
    # LR=$4                         # learning rate
    # BS=$5                         # training batch size
    # EP=$6                         # training epoches
    # N=$7                          # how many templates to use for training, -1 for using all, default = 2
    # OUTPUT_DIR=$8                 # output dir
    # WANDB_RUN_NAME=$9             # (default disabled) for inspecting training progress 
    # CONCAT_AUG_NUM=${10}          # num of augmentations (0 for No Aug, >1 for Concat)
    # CUDA_VISIBLE_DEVICES=${11}    # CUDA devices
    # --main_process_port ${12}     # for accelerate config 
    # --num_processes ${13}         # for accelerate config
    # --gradient_accumulation_step  # default = 1
####################################

bash ./training/run_mixture_baseline_offline.sh \
"${MIXTURE}" \
"${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
"google/t5-large-lm-adapt" \
0.0001 \
1 \
10 \
2 \
"${OUTPUT_SRC_ROOT}/p3_finetuning/concat_large" \
"not_using_wandb" \
5 \
${CUDA_VISIBLE_DEVICES} \
${MAIN_PROCESS_PORT} \
${NUM_PROCESSES} \
1
echo "done t5-large concat 5 aug"

### eval arg format ###
#####
    # DATASET_DIR_NAME=$1     # dataset names to be evaluated on  
    # DATASETS_DIR=$2         # dir containing the huggingface datasets  
    # OUTPUT_DIR=$3           # output dir
    # MODEL_PATH=$4           # trained model checkpoint dir  
    # CUDA_VISIBLE_DEVICES=$5 # CUDA devices
    # --main_process_port $6  # for accelerate config 
    # --num_processes $7      # for accelerate config
    # eval_bs $8              # eval batch size
#####

### eval ###

for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "hellaswag"
do
    bash ./eval/run_eval_finetuned_mixture_baseline.sh \
    ${TASK_NAME} \
    "${DATA_ROOT}/p3_c4_document_level_chosen_examples/30aug" \
    "${OUTPUT_SRC_ROOT}/eval/concat_large/${TASK_NAME}" \
    "${OUTPUT_SRC_ROOT}/p3_finetuning/concat_large" \
    ${CUDA_VISIBLE_DEVICES} \
    ${MAIN_PROCESS_PORT} \
    ${NUM_PROCESSES} \
    1 # eval batch size
done