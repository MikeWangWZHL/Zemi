import subprocess
import os


TASKS = [
    "cos_e/v1.11",
    "openbookqa/main",
    "piqa",
    "cosmos_qa",
    "dream",
    "qasc",
    "sciq",
    "quartz",
    "social_i_qa",
    "wiqa"
]

_input_dir = '/data/p3_c4_document_level' # raw retrieved datset dir
_output_dir = '/data/p3_c4_document_level_chosen_examples' # processed dataset dir
aug_k = 30 # maximum num of augmentations per instance


for task in TASKS:
    if '/' in task:
        dataset_name, dataset_config_name = task.split('/')
        input_dir = f'{_input_dir}/{task.replace("/", "_")}'
        output_dir = f'{_output_dir}/{task.replace("/", "_")}'
        cmd = f'python _process_p3_C4_retrieval_datasets.py --k {aug_k} --dataset_name {dataset_name} --dataset_config_name {dataset_config_name} --input_dir {input_dir} --output_dir {output_dir} --num_proc 16'
    else:
        dataset_name = task
        input_dir = f'{_input_dir}/{task.replace("/", "_")}'
        output_dir = f'{_output_dir}/{task.replace("/", "_")}'
        cmd = f'python _process_p3_C4_retrieval_datasets.py --k {aug_k} --dataset_name {dataset_name} --input_dir {input_dir} --output_dir {output_dir} --num_proc 16'

    if not os.path.isdir(input_dir) or os.stat(input_dir).st_size == 0:
        print(f'empty/nonexistent input dir, skip: {task}')
        continue
    if os.path.isdir(output_dir) and os.stat(output_dir).st_size > 0:
        print(f'nonempty output dir exists, skip: {task}')
        continue
    print(cmd + '\n')

    try:
        subprocess.call(cmd, shell=True)
    except Exception as e:
        print('unexpected error')
        print(e)