from datasets import load_from_disk
import random
import os
from glob import glob

random.seed(42)

def shuffle_within_task():
    input_root = "/data/p3_c4_document_level_chosen_examples/30aug"
    output_root = "/data/p3_c4_document_level_chosen_examples/30aug_shuffled"
    num_proc = 1
    task_names = [os.path.basename(item) for item in glob(os.path.join(input_root,"*"))]
    print(task_names)
    
    for task_name in task_names:
        for ds_path in glob(os.path.join(input_root,task_name,"*")):
            print(ds_path)
            ds = load_from_disk(ds_path)

            def get_shuffled_augmentation(example, idx):
                num_of_instance = len(ds_split)
                cand_indices = list(range(num_of_instance))
                cand_indices.remove(idx)
                picked_idx = random.choice(cand_indices)
                # print("original examples:", example["chosen_examples"][0])
                # print("original examples:", example["chosen_examples"][1])
                assert picked_idx != idx
                example["chosen_examples"] = ds_split[picked_idx]["chosen_examples"]
                # print("shuffled examples:", example["chosen_examples"][0])
                # print("shuffled examples:", example["chosen_examples"][1])
                return example

            for key in ["train", "validation", "test"]:
                if key not in ds:
                    continue
                ds_split = ds[key]
                ds[key] = ds_split.map(
                    get_shuffled_augmentation, 
                    with_indices=True,
                    num_proc=num_proc
                )

            os.makedirs(os.path.join(output_root, task_name), exist_ok=True)
            ds.save_to_disk(os.path.join(output_root, task_name, os.path.basename(ds_path)))

def shuffle_cross_task():
    input_root = "data/p3_c4_document_level_chosen_examples/30aug"
    output_root = "data/p3_c4_document_level_chosen_examples/30aug_cross_shuffled"
    num_proc = 1
    task_names = [os.path.basename(item) for item in glob(os.path.join(input_root,"*"))]
    print(task_names)
    
    loaded_datasets_per_task = {}
    for task_name in task_names:
        one_template_path = glob(os.path.join(input_root,task_name,"*"))[0]
        loaded_datasets_per_task[task_name] = load_from_disk(one_template_path)
    
    for task_name in task_names:

        other_task_names = [t for t in task_names if t != task_name]
        print(f"cur task: {task_name}")
        print(f"other tasks: {other_task_names}")
        
        for ds_path in glob(os.path.join(input_root,task_name,"*")):
            print(ds_path)
            ds = load_from_disk(ds_path)

            def get_shuffled_augmentation(example, idx):
                # print("cur task:",task_name)
                picked_task = random.choice(other_task_names)
                # print('picked_task:',picked_task)
                if key == "test" and key not in loaded_datasets_per_task[picked_task]:
                    other_task_ds = loaded_datasets_per_task[picked_task]["validation"]
                else:
                    if key not in loaded_datasets_per_task[picked_task]:
                        other_task_ds = loaded_datasets_per_task[picked_task]["train"]
                    else:
                        other_task_ds = loaded_datasets_per_task[picked_task][key]


                # print('split:',key)
                # print('picked_task len:',len(other_task_ds))
                picked_sample = random.choice(other_task_ds)
                # print("picked_sample:",picked_sample)
                # print("\n\n\n")
                # print("original examples:", example["chosen_examples"][0])
                # print("original examples:", example["chosen_examples"][1])
                example["chosen_examples"] = picked_sample["chosen_examples"]
                # print("shuffled examples:", example["chosen_examples"][0])
                # print("shuffled examples:", example["chosen_examples"][1])
                return example

            for key in ["train", "validation", "test"]:
                if key not in ds:
                    continue
                ds_split = ds[key]
                ds[key] = ds_split.map(
                    get_shuffled_augmentation, 
                    with_indices=True,
                    num_proc=num_proc
                )

            os.makedirs(os.path.join(output_root, task_name), exist_ok=True)
            ds.save_to_disk(os.path.join(output_root, task_name, os.path.basename(ds_path)))

if __name__ == "__main__":
    ### shuffle within task ###
    # shuffle_within_task()

    ### shuffle cross task ###
    shuffle_cross_task()