import argparse
import logging
import os
import random
import json
from glob import glob

import datasets
import torch
from datasets import (
    load_dataset,
    load_metric,
    load_from_disk,
    concatenate_datasets
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
)
from promptsource.templates import DatasetTemplates

# from t0.data_collator import DataCollatorForMultipleChoice
# use custom data collator to add indices
# import sys
# src_root = os.path.dirname(os.getcwd())
# TRAINING_DIR = os.path.join(src_root, 'training') # TODO
# sys.path.insert(1, TRAINING_DIR)

from data_collator import DataCollatorForMultipleChoice
# from t0.model import ModelBase


logger = logging.getLogger(__name__)

def get_dataset_name_2_if_origianl():
    task_2_templates = json.load(open("/cephfs/user/mikeeewang/summer_22/workspace/data/bigscience_P3_task_2_templates.json"))
    dataset_name_2_if_origianl = {}
    for key,value in task_2_templates.items():
        if 'original_dataset_name' in value:
            for d in value['original_dataset_name']:
                dataset_name_2_if_origianl[d] = True
        if 'omit_dataset_name' in value:
            for d in value['omit_dataset_name']:
                dataset_name_2_if_origianl[d] = False
    return dataset_name_2_if_origianl



def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--step_num",
        type=int,
        help="step num of the checkpoint; set to -1 if using all steps",
        default=-1,
        required=False,
    )
    parser.add_argument(
        "--concat_aug_num",
        type=int,
        help="if set to larger than 0, concat the number of chosen examples to input and then truncate to the max input length",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--filter_dataset",
        action="store_true",
        help="if doing filtering (filter out non-original) on datasets based on the template name",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--add_special_token",
    #     action="store_true",
    #     help="if set add_special_token to be true in tokenizer",
    # )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--error_analysis_dir",
        type=str,
        default="/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/error_analysis",
        help="Where to store the error analysis files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "-pdp",
        "--processed_dataset_paths",
        type=str,
        nargs='+',
        default=[],
        help=""
    )
    parser.add_argument(
        "--eval_average",
        action="store_true",
        help="If passed, will only evaluate average score.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    if args.step_num == -1 and os.path.exists(args.model_name_or_path):
        step_names = sorted([os.path.basename(p) for p in glob(os.path.join(args.model_name_or_path, "step_*"))])
    else:
        step_names = sorted([f'step_{args.step_num}'])
    

    # load args:
    model_args = json.load(open(os.path.join(args.model_name_or_path, "args.json")))
    args.concat_aug_num = model_args['concat_aug_num']
    args.max_length = model_args['max_length']
    args.target_max_length = model_args['target_max_length']


    logger.info(step_names)
    logger.info(f"concat {args.concat_aug_num} augmentation at input")
    logger.info(f"input max_length = {args.max_length}")
    logger.info(f"target max_length = {args.target_max_length}")
    # logger.info(f"processed_dataset_paths = {args.processed_dataset_paths}")

    for step_name in step_names:
        if not os.path.exists(args.model_name_or_path):
            model_name_or_path = args.model_name_or_path # load from pretrained (w/o finetuning)
        else:
            model_name_or_path = os.path.join(args.model_name_or_path, step_name)
            logger.info(f"evaluating checkpoint: {model_name_or_path}")
        
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name)
        elif model_name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path)
        else:
            raise ValueError(
                "Either `args.config_name` or `model_name_or_path` should be provided."
            )

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args['model_name_or_path'], use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if tokenizer.pad_token is None:
            for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
                if token is not None:
                    tokenizer.pad_token = token
            if tokenizer.pad_token is None:
                raise ValueError("Please define a pad token id.")

        if model_name_or_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config
            )
        else:
            raise ValueError(
                "`model_name_or_path` should be provided."
            )

        # Preprocessing the datasets.
        padding = "max_length" if args.pad_to_max_length else False

        def process_eval(examples, indices):
            bs = len(examples['inputs_pretokenized'])

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                # input, target = template.apply(ex)
                input = ex['inputs_pretokenized'].strip()
                
                if args.concat_aug_num > 0:
                    for aug in ex['chosen_examples'][:args.concat_aug_num]:
                        input = input + '\n' + aug['inputs_pretokenized'].strip()
                
                target = ex['targets_pretokenized'].strip()
                # ex_answer_choices = template.get_answer_choices_list(ex)
                ex_answer_choices = ex['answer_choices']
                try:
                    assert target in ex_answer_choices
                except AssertionError:
                    logger.warning(f'unmatched target and answer_choices: `{target}` `{ex_answer_choices}`. Using the first one in the answer_choices')
                    target = ex_answer_choices[0]
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)
            # logger.info(input_texts[0])
            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False
                # add_special_tokens=args.add_special_token,
            )
            # logger.info(tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0]))
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True
                )
                for ans_choi in answer_choices_texts
            ]

            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]
            features["indices"] = indices
            return features


        # mapping to check if original task
        if args.filter_dataset:
            dataset_name_2_if_origianl = get_dataset_name_2_if_origianl()

        raw_eval_datasets = []
        for dataset_path in args.processed_dataset_paths:
            dataset = os.path.basename(dataset_path)
            if args.filter_dataset:
                if dataset not in dataset_name_2_if_origianl:
                    logger.info('ERROR: unseen dataset:',dataset)
                    quit()
                if not dataset_name_2_if_origianl[dataset]:
                    logger.info(f'!!! skip non-original:{dataset}')
                    continue
            if '_score_eval' in dataset:
                continue
            logger.info(f'loading dataset: {dataset}')
            raw_dataset = load_from_disk(dataset_path)
            try:
                raw_eval_dataset = raw_dataset['validation']
            except KeyError:
                logger.warning(f'no validation set, skip {dataset}')
                continue
            if 'answer_choices' in raw_eval_dataset.features:
                raw_eval_datasets.append(raw_eval_dataset)
            else:
                logger.warning(f'no `answer_choices`, skip {dataset}')
        if args.eval_average:
            raw_eval_datasets = [concatenate_datasets(raw_eval_datasets)]
            args.processed_dataset_paths = ['average']
        for dataset_path, raw_eval_dataset in zip(args.processed_dataset_paths, raw_eval_datasets):
            column_names = raw_eval_dataset.column_names if raw_eval_dataset else None
            eval_dataset = raw_eval_dataset.map(
                process_eval,
                batched=True,
                remove_columns=column_names,
                with_indices=True
            )
            # # Log a few random samples from the eval set:
            # for index in random.sample(range(len(eval_dataset)), 3):
            #     logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

            # DataLoaders creation:
            if args.pad_to_max_length:
                # If padding was already done ot max length, we use the default data collator that will just convert everything
                # to tensors.
                data_collator = default_data_collator
            else:
                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                data_collator = DataCollatorForMultipleChoice(
                    tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                )

            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

            # Use the device given by the `accelerator` object.
            if not args.parallelize:
                model.to(accelerator.device)

            # Prepare everything with our `accelerator`.
            eval_dataloader = accelerator.prepare(eval_dataloader)

            # Metrics
            metric = load_metric("accuracy")

            # Eval
            total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

            logger.info("***** Running evaluation *****")
            logger.info(f"  NOTE: if add_special_tokens = False")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
            logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

            model.eval()
            # for batch in eval_dataloader:
            #     with torch.no_grad():
            #         predictions = model(batch)

            #     metric.add_batch(
            #         predictions=accelerator.gather(predictions),
            #         references=accelerator.gather(batch["targets"]),
            #     )

            #     progress_bar.update(1)

            # for error analysis
            all_predictions = []
            all_targets = []
            all_indices = []

            for batch in eval_dataloader:
                model_inputs = {
                    k: batch[k]
                    for k in ["input_ids", "attention_mask", "labels"]
                }
                with torch.no_grad():
                    logits = model(**model_inputs).logits
                masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
                seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
                seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
                # This reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
                seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
                predictions = seq_log_prob.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["targets"]),
                )

                # for error analysis
                all_predictions += [int(item) for item in accelerator.gather(predictions).detach().cpu().numpy()]
                all_targets += [int(item) for item in accelerator.gather(batch["targets"]).detach().cpu().numpy()]
                all_indices += [int(item) for item in accelerator.gather(batch["indices"]).detach().cpu().numpy()]

                progress_bar.update(1)

            # logger.info(all_predictions)
            # logger.info(all_targets)
            # logger.info(all_indices)
            # assert len(all_predictions) == len(all_targets) == len(all_indices)
            # logger.info(len(all_predictions))

            eval_metric = metric.compute()
            accelerator.print(f"Result: {os.path.basename(dataset_path)} {eval_metric}")
            results = {
                "eval_file": dataset_path,
                "evaluation": eval_metric
            }
            if accelerator.is_main_process:
                if args.output_dir is not None:
                    step_output_dir = os.path.join(args.output_dir, step_name)
                    os.makedirs(step_output_dir, exist_ok=True)
                    output_path = os.path.join(step_output_dir, os.path.basename(dataset_path))
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=4)

                ## output error analysis
                # output_error_analysis_dir = os.path.join(args.error_analysis_dir, os.path.basename(args.output_dir), os.path.basename(dataset_path))
                # os.makedirs(output_error_analysis_dir,exist_ok=True)
                # with open(os.path.join(output_error_analysis_dir, 'prediction_target_indices.json'), 'w') as out:
                #     dump_object = {
                #         "predictions":all_predictions,
                #         "targets":all_targets,
                #         "indices":all_indices
                #     }
                #     json.dump(dump_object, out)
                # raw_eval_dataset.save_to_disk(os.path.join(output_error_analysis_dir, 'dataset'))
                # print(raw_eval_dataset)
                # print('save raw dataset to disk for error analysis:',os.path.join(output_error_analysis_dir, 'dataset'))


if __name__ == "__main__":
    main()
