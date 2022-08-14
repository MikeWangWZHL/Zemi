import argparse
import logging
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import csv
import math
import json
from glob import glob
from copy import deepcopy

import datasets
import torch
from datasets import (
    load_dataset,
    load_metric,
    load_from_disk,
    concatenate_datasets,
    Dataset
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    Adafactor,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from promptsource.templates import DatasetTemplates


logger = logging.getLogger(__name__)


os.environ['TOKENIZERS_PARALLELISM'] = "false"

def log_param_size(model, logger):
    tot_params = sum(p.numel() for p in model.parameters())
    logger.info(f"total params: {tot_params}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multitask Fine-tuning baseline")

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Where to store the results CSV and (TODO) optionally the final model."
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "Path to pretrained model or model identifier from huggingface.co/models. "
            "The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`"
        ),
    )
    parser.add_argument(
        "--concat_aug_num",
        type=int,
        help="if set to larger than 0, concat the number of chosen examples to input and then truncate to the max input length",
        default=0,
        required=False,
    )
    parser.add_argument(
        "-pa",
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "-eb",
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader. Will be multiplied by the number of answer choices.",
    )
    parser.add_argument(
        "-tb",
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "-ns",
        "--num_shots",
        type=int,
        default=None,
        help="Number of training examples for few-shot learning. Default is None, which uses the entire train set.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "-ep",
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "-ms",
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "-ga",
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "-ie",
        "--input_eos",
        action="store_true",
        help=(
            "T0 was trained without EOS in its input sequences, which is the default in this script."
            "However, T5 was pretrained with EOS in its input sequences. See README for more info."
        ),
    )
    parser.add_argument(
        "-db",
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "-ftd",
        "--filter_dataset",
        action="store_true",
        help="if doing filtering (filter out non-original) on datasets based on the template name",
    )
    parser.add_argument(
        "-wb",
        "--wandb_proj",
        type=str,
        default=None,
        help="Project name for Weights & Biases. By default, W&B is disabled.",
    )
    parser.add_argument(
        "-wbrn",
        "--wandb_run_name",
        type=str,
        default=None,
        help="Run name for Weights & Biases",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=42,
        help="Especially important for few-shot example sampling.",
    )
    parser.add_argument(
        "-cf",
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "-tk",
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "-il",
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "-tl",
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "-pml",
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "-st",
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the AdamW optimizer."
    )
    parser.add_argument(
        "-ls",
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help='The scheduler type to use (choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]).',
    )
    parser.add_argument(
        "-ws",
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "-wr",
        "--warmup_ratio",
        type=float,
        default=0,
        help="Ratio for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model at each epoch.",
    )
    parser.add_argument(
        "--use_processed_dataset",
        action="store_true",
        help="Use pre-processed dataset",
    )
    parser.add_argument(
        "-ddn",
        "--dataset_dir_names",
        type=str,
        nargs='+',
        default=[],
        help="dataset dir names in dataset root"
    )
    parser.add_argument(
        "-sample_n",
        "--sample_n",
        type=int,
        help="sample n templates per dataset",
        required=True
    )
    parser.add_argument(
        "-dr",
        "--dataset_root_path",
        type=str,
        help="dataset root path"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Evaluate validation set(s) instead of computing LM loss. (all examples shoudl have the same number of choices).",
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default=None,
        help=(
            "load saved model dir"
        ),
    )
    parser.add_argument(
        "--saved_model_step",
        type=int,
        default=None,
        help=(
            "load saved model checkpoint step"
        ),
    )

    args = parser.parse_args()
    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id]*(max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0]*(max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


def get_dataset_name_2_if_origianl():
    task_2_templates = json.load(open("/data1/mikeeewang/data/bigscience_P3_task_2_templates.json"))
    dataset_name_2_if_origianl = {}
    for key,value in task_2_templates.items():
        if 'original_dataset_name' in value:
            for d in value['original_dataset_name']:
                dataset_name_2_if_origianl[d] = True
        if 'omit_dataset_name' in value:
            for d in value['omit_dataset_name']:
                dataset_name_2_if_origianl[d] = False
    return dataset_name_2_if_origianl


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    # max length config
    logger.info(f"input max_length: {args.max_length}")
    logger.info(f"target max_length: {args.target_max_length}")


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
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
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


    # mapping to check if original task
    if args.filter_dataset:
        dataset_name_2_if_origianl = get_dataset_name_2_if_origianl()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently

    sample_n = args.sample_n # select
    processed_dataset_paths = []
    for dataset_dir_name in args.dataset_dir_names:
        p = os.path.join(args.dataset_root_path, dataset_dir_name)
        all_paths = sorted(glob(os.path.join(p, '*')))
        if sample_n < 0 or sample_n > len(all_paths):
            sample_n = len(all_paths)
            logger.info(f"use all templates for {dataset_dir_name}")
        processed_dataset_paths += random.sample(all_paths, sample_n)
    logger.info(processed_dataset_paths)

    # output dataset list
    with open(os.path.join(args.output_dir, "dataset_mixture_config.json"), 'w') as o:
        sampled_datasets = {
            "sample_n":sample_n,
            "sampled_list":processed_dataset_paths,
            "dataset_dir_names":args.dataset_dir_names,
            "dataset_root_path":args.dataset_root_path
        }
        json.dump(sampled_datasets, o, indent=4)

    # load datasets
    raw_train_datasets = []
    raw_eval_datasets = []
    for dataset_path in processed_dataset_paths:
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
        if os.path.isdir(f'{dataset_path}_sampled'):
            logger.info(f'  use sampled dataset: {dataset}')
            raw_train_dataset = load_from_disk(f'{dataset_path}_sampled')
        else:
            raw_train_dataset = raw_dataset['train']
        if args.num_shots is not None:
            sample_indices = random.sample(
                range(0, len(raw_train_dataset)), k=args.num_shots)
            raw_train_dataset = raw_train_dataset.select(sample_indices)
        raw_train_datasets.append(raw_train_dataset)
        try:
            raw_eval_datasets.append(raw_dataset['validation'])
        except KeyError:
            if 'test' in raw_dataset:
                raw_eval_datasets.append(raw_dataset['test'])

    raw_train_dataset = concatenate_datasets(raw_train_datasets)
    raw_eval_dataset = concatenate_datasets(raw_eval_datasets)

    # Trim a number of evaluation examples
    if args.debug:
        raw_train_dataset = raw_train_dataset.select(
            range(min(100, len(raw_train_dataset)))
        )
        raw_eval_dataset = raw_eval_dataset.select(
            range(min(100, len(raw_eval_dataset)))
        )

    column_names = raw_eval_dataset.column_names if raw_eval_dataset else None

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    logger.info(f"INFO: tokenizer padding side:{tokenizer.padding_side}")

    # load model
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)
    model = accelerator.prepare(model)
    log_param_size(model, logger)

    
    # save args
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        out_args = deepcopy(vars(args))
        out_args['lr_scheduler_type'] = str(out_args['lr_scheduler_type'])
        json.dump(out_args, f, indent=4)

    assert not isinstance(vars(args), str)

    if args.saved_model_dir is not None:
        assert args.saved_model_step is not None
        ckpt_path = os.path.join(args.saved_model_dir, f"step_{args.saved_model_step}")
        logger.info(f"ckpt load from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'trainable param: {name}')

    logger.info(f"concat {args.concat_aug_num} augmentation at input")

    def process_train(examples):
        bs = len(examples['inputs_pretokenized'])

        input_texts = [] # list of strings
        target_texts = [] # list of strings

        for i in range(bs):
            if args.concat_aug_num > 0:
                keys = ["inputs_pretokenized", "chosen_examples", "targets_pretokenized"]
            else:
                keys = ["inputs_pretokenized", "targets_pretokenized"]

            ex = {
                k: examples[k][i] for k in keys
            }

            input_text = ex['inputs_pretokenized'].strip()
            if args.concat_aug_num > 0:
                for aug in ex['chosen_examples'][:args.concat_aug_num]:
                    input_text = input_text + '\n' + aug['inputs_pretokenized'].strip()

            target_text = ex['targets_pretokenized'].strip()

            input_texts.append(input_text)
            target_texts.append(target_text)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False
        )

        tokenized_targets = tokenizer(
            target_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=True
        )

        features = {
            "input_ids":tokenized_inputs.input_ids,
            "attention_mask":tokenized_inputs.attention_mask,
            "labels":tokenized_targets.input_ids
        }
        return features

    padding = "max_length" if args.pad_to_max_length else False
    def process_eval(examples):
        bs = len(examples['inputs'])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }
            input = ex['inputs_pretokenized']
            if args.concat_aug_num > 0:
                for aug in ex['chosen_examples'][:args.concat_aug_num]:
                    input = input + '\n' + aug['inputs_pretokenized'].strip()
            target = ex['targets_pretokenized'].strip("\n")
            ex_answer_choices = ex['answer_choices']
            assert target in ex_answer_choices
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        )

        tokenized_targets = [
            tokenizer(
                ans_choi,
                padding=True,
                max_length=args.target_max_length,
                truncation=True,
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

        return features


    if args.do_eval:
        eval_dataset_processor = process_eval
    else:
        eval_dataset_processor = process_train
    with accelerator.main_process_first():
        eval_dataset = raw_eval_dataset.map(
            eval_dataset_processor,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=column_names
        )
        train_dataset = raw_train_dataset.map(
            process_train,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=column_names
        )


    # Log a few random examples:
    if args.num_shots is not None:
        for index in random.sample(range(len(train_dataset)), min(3, args.num_shots)):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    else:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")



    # DataLoaders creation:
    train_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collator,
        batch_size=args.per_device_train_batch_size
    )
    if args.do_eval:
        eval_collator = DataCollatorForMultipleChoice(
            tokenizer,
            pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    else:
        eval_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_collator,
        batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.warmup_ratio and args.num_warmup_steps == 0:
        args.num_warmup_steps = math.ceil(args.warmup_ratio * args.max_train_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.parallelize:
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 1, "You need at least 2 GPUs to use `model.parallelize()`."
        model.parallelize()
        optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            optimizer, train_dataloader, eval_dataloader)
    else:
        optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, eval_dataloader, lr_scheduler)


    # wandb.ai
    if args.wandb_proj and accelerator.is_main_process:
        import wandb
        extra_metadata = {}
        run_config = vars(args)
        run_config.update(extra_metadata)
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else os.path.basename(args.output_dir)
        wandb.init(
            project=args.wandb_proj,
            config=run_config,
            name=wandb_run_name,
            # reinit=True, # uncomment if running multiple runs in one script
        )

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f'  Num warmup steps: {args.num_warmup_steps}')
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(math.ceil(args.max_train_steps/accelerator.num_processes)),
        disable=not accelerator.is_local_main_process
    )
    global_steps = 0

    result_table = []
    for epoch in range(1, args.num_train_epochs+1):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                loss = loss.item()
                # if accelerator.is_main_process:
                #     tqdm.write(f"epoch = {epoch}, step = {global_steps}, loss = {loss}")
                if args.wandb_proj and accelerator.is_main_process:
                    wandb.log({"loss": loss}, step=global_steps)
                    wandb.log({"last_lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)

            if global_steps >= args.max_train_steps:
                break

        # Evaluate every epoch
        total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
        # Only show the progress bar once on each machine.
        # Commented out to avoid nested pbar mess:
        # progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

        if args.do_eval:
            # Metrics
            metric = load_metric("accuracy")
            model.eval()
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
                seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)  # TODO: this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
                predictions = seq_log_prob.argmax(dim=-1)

                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["targets"]),
                )

                # progress_bar.update(1)

            eval_metric = metric.compute()
            score = eval_metric["accuracy"]  # TODO support other metrics; currently hardcoded at load_metric() anyway
            accelerator.print(f"Accuracy: {score}")
            result_table.append({
                "epoch": epoch,
                "step": global_steps,
                "metric": 'accuracy',
                "score": score,
            })
            if args.wandb_proj and accelerator.is_main_process:
                wandb.log({"accuracy": score}, step=global_steps)

        else:
            model.eval()
            dev_loss = []
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    loss = loss.item()
                    dev_loss.append(loss)
            dev_loss = sum(dev_loss) / len(dev_loss)
            result_table.append({
                "epoch": epoch,
                "step": global_steps,
                "dev_loss": dev_loss,
            })
            if accelerator.is_main_process:
                tqdm.write(f"epoch = {epoch}, step = {global_steps}, dev_loss = {dev_loss}")
            if args.wandb_proj and accelerator.is_main_process:
                wandb.log({"dev_loss": dev_loss}, step=global_steps)

        if accelerator.is_main_process:
            if args.output_dir is not None and args.save_model:
                try:
                    model.module.save_pretrained(f"{args.output_dir}/step_{global_steps}")
                except AttributeError:
                    model.save_pretrained(f"{args.output_dir}/step_{global_steps}")
    # End training loop

    if accelerator.is_main_process:
        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
                writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
                writer.writeheader()
                writer.writerows(result_table)
        if args.wandb_proj:
            wandb.finish()


if __name__ == "__main__":
    main()