# Quick Start Scripts

## Code Description
The shell scripts started with `run_*` in `zemi/training` and `zemi/eval` are lowest level scripts that will call training and evaluation `.py` files. For conveniently run multiple experiments, we provide higher level scripts which will call both training and evalution lower level shell scripts, such as `zemi_base.sh`. The comments for each positional arguments can be found in each of the higher level scripts, such as `zemi_base.sh`. For Zemi and FiD models, we will further include a config file from `perceiver_configs/` which is used for config the model architecture, such as the number of augmentation, latent size, etc (detailed in the following section)

## Perceiver Config Documentation
The json file in `perceiver_configs/` directory contains the main configuration for Zemi and FiD model architecture. The following table shows the description of each fields. 

| Field Name      | Description |
| ----------- | ----------- |
| dim | hidden dim; 768 for base, 1024 for large |
| num_latents | size of the latent query vector; default = 64 |
| num_aug_sources | maximum num of augmentations to be considered for each instance |
| depth | num of layers of the perceiver resampler, default = 1 |
| heads | num of heads of the perceiver resampler, default = 8 |
| dim_head | dim for each head, default = 64 |
| ff_mult | FFN layer scaling ratio, default = 2 |
| xattn_ff_mult | Cross-attn FFN layer scaling ratio, default = 4 |
| freeze_lm | If freeze the backbone language model, default = False |
| cross_attn_every | * for future extension, can be ignored, default = 1 |
| only_attend_immediate_media | * for future extension, can be ignored, default = False |
| num_xattn_layers | * for future extension, can be ignoredm default = 1  |

## Examples

### No Aug baseline 
- base: `bash ./training/no_aug_base.sh`
- large: `bash ./training/no_aug_large.sh`
### Concat baseline
- base: `bash ./training/concat_base.sh`
- large: `bash ./training/concat_large.sh`
### FiD baseline
- base: `bash ./training/fid_base.sh`
- large: `bash ./training/fid_large.sh`
### Zemi
- base: `bash ./training/zemi_base.sh`
- large: `bash ./training/zemi_large.sh`