#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


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
                    if k not in ["targets","indices"]
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
        batch["indices"] = torch.tensor([f.pop("indices") for f in features])
        return batch

@dataclass
class DataCollatorForMultipleChoicePromptGenerator:
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
                    if k not in ["targets","indices"]
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

        # Pad the prompt inputs because it's not padded automatically
        max_prompt_input_length = max([len(elem["prompt_input_ids"]) for elem in flattened_features])
        batch["prompt_input_ids"] = [
            l + [self.tokenizer.pad_token_id]*(max_prompt_input_length - len(l))
            for l in [elem["prompt_input_ids"] for elem in flattened_features]
        ]
        batch["prompt_attention_mask"] = [
            m + [0]*(max_prompt_input_length - len(m))
            for m in [elem["prompt_attention_mask"] for elem in flattened_features]
        ]
        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }
        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        batch["indices"] = torch.tensor([f.pop("indices") for f in features])
        return batch

@dataclass
class DataCollatorForMultipleChoiceXattn:
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
                    if k not in ["targets","indices"]
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]

        flattened_features = list(chain(*flattened_features))

        ## pad augmentation
        max_aug_num = 0
        max_aug_length = 0
        for feature in flattened_features:
            max_aug_num = max(max_aug_num, len(feature['aug_input_ids']))
            for item in feature['aug_input_ids']:
                max_aug_length = max(max_aug_length, len(item))

        padding_side = self.tokenizer.padding_side
        for feature in flattened_features:
            ### pad the augmentation
            aug_input_ids_per_instance = feature['aug_input_ids']
            aug_attention_mask_per_instance = feature['aug_attention_mask']
            padded_aug_input_ids_per_instance = []
            padded_aug_attention_mask_per_instance = []
            aug_exist_idx = [] # for input as aug_exist_idx in gatedcrossattention: indicating which of the augmentation positions are used
            for i in range(len(aug_input_ids_per_instance)):
                ids = aug_input_ids_per_instance[i]
                mask = aug_attention_mask_per_instance[i]
                ids_remainder = [self.tokenizer.pad_token_id] * (max_aug_length - len(ids))
                mask_remainder = [0] * (max_aug_length - len(ids))
                ids = (
                    ids + ids_remainder if padding_side == "right" else ids_remainder + ids
                )
                mask = (
                    mask + mask_remainder if padding_side == "right" else mask_remainder + mask
                )
                padded_aug_input_ids_per_instance.append(ids)
                padded_aug_attention_mask_per_instance.append(mask)
                aug_exist_idx.append(1)
            
            while len(padded_aug_input_ids_per_instance) < max_aug_num:
                padded_aug_input_ids_per_instance.append([self.tokenizer.pad_token_id] * max_aug_length)
                padded_aug_attention_mask_per_instance.append([0] * max_aug_length)
                aug_exist_idx.append(0)
            
            # pad to same num of augmentation
            feature['aug_input_ids'] = padded_aug_input_ids_per_instance
            feature['aug_attention_mask'] = padded_aug_attention_mask_per_instance
            feature['aug_exist_idx'] = aug_exist_idx
        
        ## using calibration
        # all dummy input should be the same length 


        ## pad all and turn into tensor
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
        batch["indices"] = torch.tensor([f.pop("indices") for f in features])
        return batch


from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

@dataclass
class DataCollatorForSeq2SeqPromptGenerator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # Pad the prompt inputs because it's not padded automatically
        prompt_input_ids = [feature["prompt_input_ids"] for feature in features] if "prompt_input_ids" in features[0].keys() else None
        if prompt_input_ids is not None:
            max_prompt_input_length = max(len(l) for l in prompt_input_ids)
            if self.pad_to_multiple_of is not None:
                max_prompt_input_length = (
                    (max_prompt_input_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder_input_ids = [self.tokenizer.pad_token_id] * (max_prompt_input_length - len(feature["prompt_input_ids"]))
                remainder_attention_mask = [0] * (max_prompt_input_length - len(feature["prompt_attention_mask"]))

                feature["prompt_input_ids"] = feature["prompt_input_ids"] + remainder_input_ids if padding_side == "right" else remainder_input_ids + feature["prompt_input_ids"]
                feature["prompt_attention_mask"] = feature["prompt_attention_mask"] + remainder_attention_mask if padding_side == "right" else remainder_attention_mask + feature["prompt_attention_mask"]

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        # # NOTE: this part is ignored for our model
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class DataCollatorForSeq2SeqXattn:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        # all_input_ids = [feature["input_ids"] for feature in features]
        # max_input_ids_length = max(len(input_ids) for input_ids in all_input_ids)

        ## get aug padding length
        max_aug_length = 0
        max_aug_num = 0
        for feature in features:
            max_aug_num = max(max_aug_num, len(feature['aug_input_ids']))
            for item in feature['aug_input_ids']:
                max_aug_length = max(max_aug_length, len(item))

        padding_side = self.tokenizer.padding_side
        for feature in features:
            ### pad the augmentation
            aug_input_ids_per_instance = feature['aug_input_ids']
            aug_attention_mask_per_instance = feature['aug_attention_mask']
            # max_aug_length = max([len(a) for a in aug_input_ids_per_instance])
            padded_aug_input_ids_per_instance = []
            padded_aug_attention_mask_per_instance = []
            aug_exist_idx = [] # for input as aug_exist_idx in gatedcrossattention: indicating which of the augmentation positions are used
            for i in range(len(aug_input_ids_per_instance)):
                ids = aug_input_ids_per_instance[i]
                mask = aug_attention_mask_per_instance[i]
                ids_remainder = [self.tokenizer.pad_token_id] * (max_aug_length - len(ids))
                mask_remainder = [0] * (max_aug_length - len(ids))
                ids = (
                    ids + ids_remainder if padding_side == "right" else ids_remainder + ids
                )
                mask = (
                    mask + mask_remainder if padding_side == "right" else mask_remainder + mask
                )
                padded_aug_input_ids_per_instance.append(ids)
                padded_aug_attention_mask_per_instance.append(mask)
                aug_exist_idx.append(1)
            
            while len(padded_aug_input_ids_per_instance) < max_aug_num:
                padded_aug_input_ids_per_instance.append([self.tokenizer.pad_token_id] * max_aug_length)
                padded_aug_attention_mask_per_instance.append([0] * max_aug_length)
                aug_exist_idx.append(0)
            
            # pad to same num of augmentation
            feature['aug_input_ids'] = padded_aug_input_ids_per_instance
            feature['aug_attention_mask'] = padded_aug_attention_mask_per_instance
            feature['aug_exist_idx'] = aug_exist_idx

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

# for opendomain qa evaluation
@dataclass
class DataCollatorForSeq2SeqXattnODQA:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        # all_input_ids = [feature["input_ids"] for feature in features]
        # max_input_ids_length = max(len(input_ids) for input_ids in all_input_ids)

        ## get aug padding length
        max_aug_length = 0
        max_aug_num = 0
        for feature in features:
            max_aug_num = max(max_aug_num, len(feature['aug_input_ids']))
            for item in feature['aug_input_ids']:
                max_aug_length = max(max_aug_length, len(item))

        ## get answers padding length
        max_ans_length = 0
        max_ans_num = 0
        for feature in features:
            max_ans_num = max(max_ans_num, len(feature['answer_input_ids']))
            for item in feature['answer_input_ids']:
                max_ans_length = max(max_ans_length, len(item))

        padding_side = self.tokenizer.padding_side
        for feature in features:
            ### pad the augmentation
            aug_input_ids_per_instance = feature['aug_input_ids']
            aug_attention_mask_per_instance = feature['aug_attention_mask']
            # max_aug_length = max([len(a) for a in aug_input_ids_per_instance])
            padded_aug_input_ids_per_instance = []
            padded_aug_attention_mask_per_instance = []
            aug_exist_idx = [] # for input as aug_exist_idx in gatedcrossattention: indicating which of the augmentation positions are used
            for i in range(len(aug_input_ids_per_instance)):
                ids = aug_input_ids_per_instance[i]
                mask = aug_attention_mask_per_instance[i]
                ids_remainder = [self.tokenizer.pad_token_id] * (max_aug_length - len(ids))
                mask_remainder = [0] * (max_aug_length - len(ids))
                ids = (
                    ids + ids_remainder if padding_side == "right" else ids_remainder + ids
                )
                mask = (
                    mask + mask_remainder if padding_side == "right" else mask_remainder + mask
                )
                padded_aug_input_ids_per_instance.append(ids)
                padded_aug_attention_mask_per_instance.append(mask)
                aug_exist_idx.append(1)

            while len(padded_aug_input_ids_per_instance) < max_aug_num:
                padded_aug_input_ids_per_instance.append([self.tokenizer.pad_token_id] * max_aug_length)
                padded_aug_attention_mask_per_instance.append([0] * max_aug_length)
                aug_exist_idx.append(0)
            
            ### pad the answers
            ans_input_ids_per_instance = feature['answer_input_ids']
            padded_ans_input_ids_per_instance = []
            for i in range(len(ans_input_ids_per_instance)):
                ids = ans_input_ids_per_instance[i]
                ids_remainder = [self.tokenizer.pad_token_id] * (max_ans_length - len(ids))
                ids = (
                    ids + ids_remainder if padding_side == "right" else ids_remainder + ids
                )
                padded_ans_input_ids_per_instance.append(ids)

            while len(padded_ans_input_ids_per_instance) < max_ans_num:
                padded_ans_input_ids_per_instance.append([self.tokenizer.pad_token_id] * max_ans_length)


            feature['aug_input_ids'] = padded_aug_input_ids_per_instance
            feature['aug_attention_mask'] = padded_aug_attention_mask_per_instance
            feature['aug_exist_idx'] = aug_exist_idx
            feature['answer_input_ids'] = padded_ans_input_ids_per_instance


        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features