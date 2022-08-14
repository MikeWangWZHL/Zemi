from multiprocessing.connection import answer_challenge
from datasets import load_from_disk
import json
from glob import glob
import os


def find_idx_get_better_and_worse(results_no_demo, results_w_demo):
    assert results_no_demo['indices'] == results_w_demo['indices']
    assert results_no_demo['targets'] == results_w_demo['targets']
    predictions_no_demo = results_no_demo['predictions']
    predictions_w_demo = results_w_demo['predictions']
    targets = results_w_demo['targets']
    indices = results_w_demo['indices']
    getting_better_indices = []
    getting_worse_indices = []
    idx_2_prediction_no_demo = {}
    idx_2_prediction_w_demo = {}
    for i in range(len(predictions_no_demo)):
        if (predictions_no_demo[i] != targets[i]) and (predictions_w_demo[i] == targets[i]):
            getting_better_indices.append(indices[i])
        elif (predictions_no_demo[i] == targets[i]) and (predictions_w_demo[i] != targets[i]):
            getting_worse_indices.append(indices[i])
        idx_2_prediction_no_demo[indices[i]] = predictions_no_demo[i]
        idx_2_prediction_w_demo[indices[i]] = predictions_w_demo[i]
    return getting_better_indices, getting_worse_indices, idx_2_prediction_no_demo, idx_2_prediction_w_demo

if __name__ == "__main__":

    # ks = [0, 1]

    error_analysis_root = '/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/error_analysis'

    dataset_name_no_demo = f"openbookqa__main__p3_subset__k-0___mulcqa_mixture_k-0_6-10_n-2"
    dataset_name_no_demo = f"piqa__none__p3_subset__k-0___mulcqa_mixture_k-0_6-10_n-2"
    # dataset_name_no_demo = f"super_glue__wic__p3_subset__k-0___mulcqa_mixture_k-0_6-10_n-2"
    
    # dataset_name_w_demo = f"mulcqa_mixture_xattn_on_encoder_instance-aug-k-3_6_27_unfreeze_lm__step-79280"
    dataset_name_w_demo = f"openbookqa_main_SharedEncoderDecoder_MultiAug_lr-0.0002_bs-2_KiC_aug_unfreeze_lm_7_2_32535"
    dataset_name_w_demo = f"piqa_SharedEncoderDecoder_MultiAug_lr-0.0001_bs-2_KiC_aug_unfreeze_lm_7_2_step_30225"
    # kic_aug_string = "_commonsense"
    kic_aug_string = "_causal"
    insert_idx = 1

    # dataset_name_w_demo = f"openbookqa__main__p3_subset__template_augmented_use_retrieved___mulcqa_mixture_template_augmented_6-12_n-2"
    # dataset_name_no_demo = f"openbookqa__main__p3_subset__k-{ks[0]}___mulcqa_mixture_k-{ks[0]}_6-10_n-2"
    # dataset_name_w_demo = f"openbookqa__main__p3_subset__k-{ks[1]}___mulcqa_mixture_k-{ks[1]}_6-10_n-2"

    # dataset_name_no_demo = f"cos_e__v1.11__p3_subset__k-{ks[0]}___mulcqa_mixture_k-{ks[0]}_6-10_n-2"
    # dataset_name_w_demo = f"cos_e__v1.11__p3_subset__k-{ks[1]}___mulcqa_mixture_k-{ks[1]}_6-10_n-2"

    # dataset_name_no_demo = f"piqa__none__p3_subset__k-{ks[0]}___mulcqa_mixture_k-{ks[0]}_6-10_n-2"
    # dataset_name_w_demo = f"piqa__none__p3_subset__k-{ks[1]}___mulcqa_mixture_k-{ks[1]}_6-10_n-2"

    ##### restrained #####

    # error_analysis_root = '/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/error_analysis_seen_training_only'
    # dataset_name_no_demo = f"openbookqa__main__p3_subset__k-{ks[0]}___mulcqa_mixture_k-{ks[0]}_6-10_n-2"
    # dataset_name_w_demo = f"openbookqa__main__p3_subset__k-{ks[1]}___mulcqa_mixture_k-{ks[1]}_6-10_n-2"

    # dataset_name_no_demo = f"piqa__none__p3_subset__k-{ks[0]}___mulcqa_mixture_k-{ks[0]}_6-10_n-2"
    # dataset_name_w_demo = f"piqa__none__p3_subset__k-{ks[1]}___mulcqa_mixture_k-{ks[1]}_6-10_n-2"


    for p in sorted(glob(os.path.join(error_analysis_root, dataset_name_no_demo, '*'))):

        p_name = os.path.basename(p)

        results_no_demo = json.load(open(os.path.join(error_analysis_root, dataset_name_no_demo, p_name, 'prediction_target_indices.json')))
        
        splitted = p_name.split("_")
        splitted.insert(insert_idx,kic_aug_string)
        p_name_w_demo = "_".join(splitted)
        print(p_name_w_demo)
        if not os.path.exists(os.path.join(error_analysis_root, dataset_name_w_demo, p_name_w_demo, 'prediction_target_indices.json')):
            continue
        results_w_demo = json.load(open(os.path.join(error_analysis_root, dataset_name_w_demo, p_name_w_demo, 'prediction_target_indices.json')))

        loaded_dataset_w_demo = load_from_disk(os.path.join(error_analysis_root, dataset_name_w_demo, p_name_w_demo, 'dataset'))

        print(loaded_dataset_w_demo)

        indices_get_better, indices_get_worse, idx_2_prediction_no_demo, idx_2_prediction_w_demo = find_idx_get_better_and_worse(results_no_demo, results_w_demo)
        print('======== indices_get_better ========')
        print(indices_get_better)
        print('======= indices_get_worse =========')
        print(indices_get_worse)

        instances_get_better = []
        for idx in indices_get_better:
            item = loaded_dataset_w_demo[idx]
            # print(item)
            # print(idx_2_prediction_w_demo[idx])
            # quit()
            instances_get_better.append(
                {
                    "inputs_pretokenized":item['inputs_pretokenized'],
                    "targets_pretokenized":item['targets_pretokenized'],
                    "chosen_examples":item["chosen_examples"],
                    "prediction":item["answer_choices"][idx_2_prediction_w_demo[idx]]
                }
            )

        instances_get_worse = []
        for idx in indices_get_worse:
            item = loaded_dataset_w_demo[idx]
            instances_get_worse.append(
                {
                    "inputs_pretokenized":item['inputs_pretokenized'],
                    "targets_pretokenized":item['targets_pretokenized'],
                    "chosen_examples":item["chosen_examples"],
                    "prediction":item["answer_choices"][idx_2_prediction_w_demo[idx]]
                }
            )

        output_instances_better_path = os.path.join(error_analysis_root, dataset_name_w_demo, p_name_w_demo, 'instances_w_demo_better.json')
        output_instances_worse_path = os.path.join(error_analysis_root, dataset_name_w_demo, p_name_w_demo, 'instances_w_demo_worse.json')

        with open(output_instances_better_path, 'w') as out:
            json.dump(instances_get_better, out, indent=4)

        with open(output_instances_worse_path, 'w') as out:
            json.dump(instances_get_worse, out, indent=4)