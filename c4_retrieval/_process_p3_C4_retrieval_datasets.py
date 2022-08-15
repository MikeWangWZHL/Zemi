import re
import subprocess
import argparse
import logging

import promptsource.templates
# TEMPLATES_FOLDER_PATH='/cephfs/user/xiaomanpan/lib/promptsource/promptsource/templates'
# promptsource.templates.TEMPLATES_FOLDER_PATH=TEMPLATES_FOLDER_PATH
# import _run_process_data


logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="chosen example number",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-t",
        "--template_name",
        type=str,
        default=None,
        help="The template/prompt name in `promptsource`.",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=42,
        help="Especially important for few-shot example sampling.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="Input arrow dump directory",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Ourput directory",
    )
    parser.add_argument(
        "-np",
        "--num_proc",
        type=int,
        default=1,
        help="Number of processors for data pre-processing"
    )
    args = parser.parse_args()

    return args


def read():
    res = {}
    input_path = '/cephfs/user/xiaomanpan/data/tmp/bigscience_P3 - overall.tsv'
    with open(input_path, 'r') as f:
        first_line = f.readline()
        assert first_line.startswith('category')
        for line in f:
            tmp = line.rstrip('\n').split('\t')
            task = tmp[1]
            dataset = tmp[2].replace('__', '_').strip('_').strip()
            num_samples = int(tmp[4]) if tmp[4] else 0
            res[dataset] = num_samples
    return res


if __name__ == '__main__':
    ns = read()

    args = parse_args()

    if args.dataset_name == 'anli':
        prompts = promptsource.templates.DatasetTemplates('anli', None)
    else:
        prompts = promptsource.templates.DatasetTemplates(
            f"{args.dataset_name}"
            if args.dataset_config_name is None
            else f"{args.dataset_name}/{args.dataset_config_name}"
        )

    for i in prompts.templates:
        template_name = prompts.templates[i].name

        if not prompts.templates[i].metadata.original_task:
            logger.info(f'is not original task, skip: {args.dataset_name} {template_name}')
            continue
        if not prompts.templates[i].metadata.choices_in_prompt:
            logger.info(f'no choices in prompt, skip: {args.dataset_name} {template_name}')
            continue

        template_name_ = template_name.replace('-', '_').replace(' ', '_').replace('/', '_').replace('___', '_')
        template_name_ = re.sub(r"[^\w\d'\s\_]+", '', template_name_).strip('_')
        if args.dataset_config_name:
            if 'ai2_arc' in args.dataset_name:
                dataset = f'{args.dataset_name}_{args.dataset_config_name.replace("-", "_")}_{template_name_}'
            else:
                dataset = f'{args.dataset_name}_{args.dataset_config_name}_{template_name_}'
        else:
            dataset = f'{args.dataset_name}_{template_name_}'
        
        print("num_samples:", ns[dataset])
        print()

        cmd = f'python process_p3_C4_retrieval_datasets.py --k {args.k} --dataset_name {args.dataset_name} --dataset_config_name {args.dataset_config_name} --input_dir {args.input_dir} --output_dir {args.output_dir} --template_name "{template_name}" --num_proc {args.num_proc} --num_samples {ns[dataset]}'
        print(cmd)
        try:
            subprocess.call(cmd, shell=True)
        except Exception as e:
            print('unexpected error')
            print(e)