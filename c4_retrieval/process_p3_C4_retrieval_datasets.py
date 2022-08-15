import os
import sys
import argparse
import logging
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import random
from collections import defaultdict

import ujson as json
from datasets import (
    load_dataset,
    load_from_disk
)
from datasets.utils import disable_progress_bar
disable_progress_bar()
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
import promptsource.templates


logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


# # load local templates
# TEMPLATES_FOLDER_PATH='/cephfs/user/xiaomanpan/lib/promptsource/promptsource/templates'
# promptsource.templates.TEMPLATES_FOLDER_PATH=TEMPLATES_FOLDER_PATH


def clean_template_name(s):
    return s.replace('/', '')


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
        required=False,
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
        required=True,
        help="The template/prompt name in `promptsource`.",
    )
    # parser.add_argument(
    #     "-st",
    #     "--use_slow_tokenizer",
    #     action="store_true",
    #     help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    # )
    # parser.add_argument(
    #     "-tk",
    #     "--tokenizer_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained tokenizer name or path if not the same as model_name",
    # )
    # parser.add_argument(
    #     "-il",
    #     "--max_length",
    #     type=int,
    #     default=1024,
    #     help=(
    #         "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
    #         " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    #     ),
    # )
    # parser.add_argument(
    #     "-tl",
    #     "--target_max_length",
    #     type=int,
    #     default=256,
    #     help="Target max length. Sequences longer than this will be truncated."
    # )
    # parser.add_argument(
    #     "-pml",
    #     "--pad_to_max_length",
    #     action="store_true",
    #     help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    # )
    # parser.add_argument(
    #     "-ie",
    #     "--input_eos",
    #     action="store_true",
    #     help=(
    #         "T0 was trained without EOS in its input sequences, which is the default in this script."
    #         "However, T5 was pretrained with EOS in its input sequences. See README for more info."
    #     ),
    # )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="chosen example number",
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
    # parser.add_argument(
    #     "-k",
    #     "--knowledge_types",
    #     type=str,
    #     nargs='+',
    #     default=[],
    #     help=""
    # )
    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=0,
        help=""
    )
    args = parser.parse_args()

    return args



# def apply_template(data, input_key, target_key, template_name):
#     question = data[input_key]
#     target = data[target_key]

#     if template_name == "question_answer":
#         input = f"Question: {question}\nAnswer:"
#     elif template_name == "answer_the_question":
#         input = f"Answer the following question. {question}"
#     elif template_name == "wondered":
#         input = f"I've always wondered: {question}"
#     elif template_name == "goal":
#         input = f"The goal is to predict an English answer string for an input English question.\nQuestion: {question}\nAnswer:"
    
#     return input, target

# task_2_generative_template_name = {
#     "cos_e_v1.11":["question_answer", "answer_the_question"], # "wondered", "goal",
#     "cosmos_qa":["context_question_description_text","description_context_question_text"],
#     "dream": ["dialogue_question_answer", "description_dialogue_question_answer"],
#     "qasc": ["facts_question_answer", "description_facts_question_answer"],
#     "quartz": ["paragraph_question_plain_concat","given_the_fact_answer_the_q_no_choice"],
#     "sciq": ["Direct Question", "Direct Question (Closed Book)"],
#     "social_i_qa": ["Generate answer", "I was wondering"],
#     "wiqa": ["what_is_the_final_step_of_the_following_process", "what_might_be_the_last_step_of_the_process"]
# }


# def get_input_target(data, template, knowledge_types):
#     input, target = template.apply(data)
#     kic = []
#     for knwl_type in data['knowledge']:
#         if knwl_type not in knowledge_types:
#             continue
#         if knwl_type == 'lexicon':
#             for key in data['knowledge'][knwl_type]['gloss']:
#                 if not data['knowledge'][knwl_type]['gloss'][key]:
#                     continue
#                 kic.append(
#                     '\n'.join(data['knowledge'][knwl_type]['gloss'][key][0]['glosses'][:5]))
#         elif knwl_type == 'causal':
#             for key in data['knowledge'][knwl_type]:
#                 for n in range(5):
#                     kic.append(data['knowledge'][knwl_type][key][n]['sent'])
#         else:
#             for key in data['knowledge'][knwl_type]:
#                 if not data['knowledge'][knwl_type][key]:
#                     continue
#                 kic.append(
#                     '\n'.join(json.loads(data['knowledge'][knwl_type][key][0]['sentence_key'])[:5]))
#     input = input + '\n' + '[KiC]\n' + '\n'.join(kic) + '\n[KiC]\n'
#     return input, target

def get_augmentation(data, k, query_keys = None):
    if query_keys is not None:
        keys = query_keys
    else:
        keys = data["retrieved_examples"].keys()
    cands = []
    for key in keys:
        cands += data["retrieved_examples"][key][:k]
    cands = sorted(cands, key = lambda x: x["score"], reverse=True)
    cands = cands[:k]
    chosen_examples = [
        {
            "inputs_pretokenized":cands[i]['text'],
            "dataset_name": "c4_retrieval",
            "template_name":"c4_retrieval",
            "idx":i, # dummy
            "inputs":[], # dummy
            "score":cands[i]['score'], # dummy
            "targets":[], # dummy
            "targets_pretokenized":"" # dummy
        } for i in range(len(cands))
    ]
    return chosen_examples


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"num of augs to be chosen: {args.k}")

    # get task name
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    if dataset_config_name:
        task_full_name = f'{dataset_name}_{dataset_config_name}'
    else:
        task_full_name = f'{dataset_name}'
    print('full task name:', task_full_name)

    # load templates
    if args.dataset_name == 'anli':
        prompts = promptsource.templates.DatasetTemplates('anli', None)
    else:
        if args.dataset_config_name == 'None':
            args.dataset_config_name = None
        prompts = promptsource.templates.DatasetTemplates(
            f"{args.dataset_name}"
            if args.dataset_config_name is None
            else f"{args.dataset_name}/{args.dataset_config_name}"
        )
    
    # assert task_full_name in task_2_generative_template_name
    # for template_name in task_2_generative_template_name[task_full_name]:
    template_name = args.template_name
    logger.info(f"template_name: {template_name}")
    template = prompts[template_name]

    def preprocess_train(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        chosen_examples = []
        answer_choices = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }

            input, target = template.apply(ex)
            chosen_examples_per_instance = get_augmentation(ex, args.k)
            ex_answer_choices = template.get_answer_choices_list(ex)
            
            input_texts.append(input)
            target_texts.append(target)
            chosen_examples.append(chosen_examples_per_instance)
            answer_choices.append(ex_answer_choices)
        
        examples['inputs_pretokenized'] = input_texts
        examples['targets_pretokenized'] = target_texts
        examples['chosen_examples'] = chosen_examples
        examples['answer_choices'] = answer_choices

        # Log a few random samples:
        for index in random.sample(range(len(input_texts)), 3):
            logger.debug(f'Template name: {template_name}')
            logger.debug(f"Sample {index} of the input texts:")
            logger.debug(f"\n{input_texts[index]}\n")
        
        return examples

    logger.info(f'loading {args.input_dir}')
    raw_dataset = load_from_disk(args.input_dir)
    logger.info(raw_dataset)
    
    if args.num_samples > 0:
        logger.info(f'sampling train split using slice: [0:{args.num_samples}]')
        raw_dataset['train'] = raw_dataset['train'].select(range(args.num_samples))

    column_names = raw_dataset['train'].column_names
    processed_dataset = raw_dataset.map(
        preprocess_train,
        batched=True,
        remove_columns=column_names,
        num_proc=args.num_proc
    )

    for index in random.sample(range(len(processed_dataset['train'])), 3):
        logger.debug(
            f"Sample {index} of the training set: {processed_dataset['train'][index]}.")

    template_name = clean_template_name(template_name)
    output_dir = f"{args.output_dir}/{task_full_name}_{template_name.replace(' ', '_')}"
    processed_dataset.save_to_disk(f'{output_dir}')


if __name__ == '__main__':
    main()