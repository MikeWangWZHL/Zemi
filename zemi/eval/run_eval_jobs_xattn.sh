### multi-knowledge eval
## input format
    # DATASET_DIR_NAME=$1
    # DATASETS_ROOT=$2 # -> DATASETS_DIR="${DATASETS_ROOT}/${DATASET_DIR_NAME}/*"
    # MODEL_PATH=$3
    # OUTPUT_DIR=$4
    # CUDA_VISIBLE_DEVICES=$5
    # --main_process_port $6
    # --num_processes $7


# MODEL_PATH_NAME="7_27_multitask_mixture_mulcqa_n_2_c4_5percent_5aug"

# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/data2/mikeeewang/data/p3_c4_document_level_chosen_examples/30aug" \
# "/data2/mikeeewang/finetune-t5/output/p3_finetuning/${MODEL_PATH_NAME}" \
# "/data2/mikeeewang/finetune-t5/output/eval/${MODEL_PATH_NAME}" \
# 4,5,6,7 \
# 20655 \
# 4





# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_19_openbook_qa_kic_all_five_knowledge_doublegated/t5-base/7_19_openbook_qa_kic_all_five_knowledge_SharedEncoderDecoder_MultiAug_DoubleGated_t5_base" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_19_openbook_qa_kic_all_five_knowledge_doublegated/t5-base/7_19_openbook_qa_kic_all_five_knowledge_SharedEncoderDecoder_MultiAug_DoubleGated_t5_base" \
# "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_knowledge_augmented/preprocessed_multiple_knowledge/all_five_knowledge/openbookqa_main/*" \
# "SharedEncoderDecoder_MultiAug_DoubleGated" \
# 0,1,2,3,4,5,6,7 \
# 20655 \
# 8


# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_26_openbook_qa_c4_5percent/t5-base/7_26_openbook_qa_c4_5percent_1aug" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_26_openbook_qa_c4_5percent/t5-base/7_26_openbook_qa_c4_5percent_1aug" \
# "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_c4_document_level_chosen_examples/10aug/openbookqa_main/*" \
# "SharedEncoderDecoder_MultiAug" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_26_openbook_qa_c4_5percent/t5-base/7_26_openbook_qa_c4_5percent_5aug" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_26_openbook_qa_c4_5percent/t5-base/7_26_openbook_qa_c4_5percent_5aug" \
# "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_c4_document_level_chosen_examples/10aug/openbookqa_main/*" \
# "SharedEncoderDecoder_MultiAug" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_26_openbook_qa_c4_5percent/t5-base/7_26_openbook_qa_c4_5percent_10aug" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_26_openbook_qa_c4_5percent/t5-base/7_26_openbook_qa_c4_5percent_10aug" \
# "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_c4_document_level_chosen_examples/10aug/openbookqa_main/*" \
# "SharedEncoderDecoder_MultiAug" \
# 4,5,6,7 \
# 20655 \
# 4



# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_27_openbook_qa_c4_5percent/t5-base/7_27_openbook_qa_c4_5percent_5aug_frozen_aug_encoder" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_27_openbook_qa_c4_5percent/t5-base/7_27_openbook_qa_c4_5percent_5aug_frozen_aug_encoder" \
# "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_c4_document_level_chosen_examples/10aug/openbookqa_main/*" \
# "SharedEncoderDecoder_MultiAug_FrozenAugEncoder" \
# 4,5,6,7 \
# 20655 \
# 4

# bash run_eval_finetuned_mixture_xattn.sh \
# "openbookqa_main" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/7_27_openbook_qa_c4_5percent/t5-base/7_27_openbook_qa_c4_5percent_10aug_frozen_aug_encoder" \
# "/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/eval/7_27_openbook_qa_c4_5percent/t5-base/7_27_openbook_qa_c4_5percent_10aug_frozen_aug_encoder" \
# "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_c4_document_level_chosen_examples/10aug/openbookqa_main/*" \
# "SharedEncoderDecoder_MultiAug_FrozenAugEncoder" \
# 4,5,6,7 \
# 20655 \
# 4


# MODEL_PATH_NAME="7_27_multitask_mixture_mulcqa_n_2_c4_5percent_5aug"

# for TASK_NAME in "hellaswag" 
# do
#     bash run_eval_finetuned_mixture_xattn.sh \
#     ${TASK_NAME} \
#     "/data1/mikeeewang/data/p3_c4_document_level_chosen_examples/30aug" \
#     "/data1/mikeeewang/finetune-t5/output/p3_finetuning/7_27_multitask_mixture_mulcqa_n_2_c4_5percent/t5-base/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_5aug" \
#     "/data1/mikeeewang/finetune-t5/output/eval/${MODEL_PATH_NAME}/${TASK_NAME}" \
#     4,5,6,7 \
#     20655 \
#     4
# done



# MODEL_PATH_NAME=""

# for TASK_NAME in "openbookqa_main" "piqa" "super_glue_wic" "super_glue_cb" "super_glue_copa" "rotten_tomatoes" "hellaswag"
# do
#     bash run_eval_finetuned_mixture_xattn.sh \
#     ${TASK_NAME} \
#     "/data1/mikeeewang/data/p3_c4_document_level_chosen_examples/30aug" \
#     "/data1/mikeeewang/finetune-t5/output/p3_finetuning/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_concat_baseline/t5-base/7_27_multitask_mixture_mulcqa_n_2_c4_5percent_concat_baseline_5aug" \
#     "/data1/mikeeewang/finetune-t5/output/eval/${MODEL_PATH_NAME}/${TASK_NAME}" \
#     4,5,6,7 \
#     20655 \
#     4
# done