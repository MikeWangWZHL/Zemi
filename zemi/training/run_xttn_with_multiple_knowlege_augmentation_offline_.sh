################
# delay start for some time
# python sleeping_foo.py --secs 18000
################

# DATASET_DIR_NAMES="openbookqa_main"
# # DATASET_ROOT="/cephfs/user/mikeeewang/summer_22/workspace/data/p3_knowledge_augmented/preprocessed_multiple_knowledge/all_five_knowledge"
# DATASET_ROOT="/cephfs/user/mikeeewang/summer_22/workspace/data/p3_knowledge_augmented/preprocessed_multiple_knowledge/causal_and_commonsense"

DATASET_DIR_NAMES=$1
DATASET_ROOT=$2

LR=$3
BS=$4
BS_EVAL=4
EP=$5
N=${14}


K=0
LATENT_SIZE=64
MAX_LENGTH=$((1024-${K}*${LATENT_SIZE}))
echo "set max length to ${MAX_LENGTH}"
AUG_MAX_LENGTH=256
TARGET_MAX_LENGTH=256


# LM_NAME="google/t5-base-lm-adapt"
# LM_NAME="t5-base"
LM_NAME=$6


# PERCEIVER_CONFIG="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/training/perceiver_configs/xattn_multi_aug_config_v1_v3_5aug.json"
PERCEIVER_CONFIG=$7

### train T5ForConditionalGenerationMultiAug ###
# MODEL_ARCHITECTURE="SharedEncoderDecoder_MultiAug"

MODEL_ARCHITECTURE=${13}

### unfreeze lm setting ###
# OUTPUT_DIR="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_17_openbook_qa_kic_causal_and_commonsense/t5-base/7_17_openbook_qa_kic_causal_and_commonsense_SharedEncoderDecoder_MultiAug"
# WANDB_RUN_NAME="7_17_openbook_qa_multiple_kic_causal_and_commonsense_SharedEncoderDecoder_MultiAug"
OUTPUT_DIR=$8
WANDB_RUN_NAME=$9

GRADIENT_ACCUMULATION_STEPS=${15}

CUDA_VISIBLE_DEVICES=${10} accelerate launch --main_process_port ${11} --num_processes ${12} multi_task_fine_tune_xattn.py \
    --model_architecture ${MODEL_ARCHITECTURE} \
    --max_length ${MAX_LENGTH} \
    --target_max_length ${TARGET_MAX_LENGTH} \
    --aug_max_length ${AUG_MAX_LENGTH} \
    --lm_name ${LM_NAME} \
    --perceiver_config ${PERCEIVER_CONFIG} \
    --dataset_dir_names ${DATASET_DIR_NAMES} \
    --dataset_root_path ${DATASET_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.25 \
    --per_device_train_batch_size ${BS} \
    --per_device_eval_batch_size ${BS_EVAL} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LR} \
    --num_proc 16 \
    --num_train_epochs ${EP} \
    --lr_scheduler_type cosine \
    --save_model \
    --sample_n ${N} \
    # --wandb_proj "summer_22_intern" \
    # --wandb_run_name ${WANDB_RUN_NAME}
