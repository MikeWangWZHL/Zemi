import subprocess
import os


_output_dir = '/data/p3_c4_document_level'

task_list_json = "./tasks/tasks_to_be_processed.json"

cmd = f'python p3_C4_document_level_retrieval.py --task_list {task_list_json} --output_dir {_output_dir} --num_proc 1'

print(cmd + '\n')

try:
    subprocess.call(cmd, shell=True)
except Exception as e:
    print('unexpected error')
    print(e)