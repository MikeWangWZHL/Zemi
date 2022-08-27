
# Code for paper Zemi 

# Prepare Datasets
Instructions on downloading preprocessed datasets and prepraring costum datasets can be found [here](./data/README.md)  

# Download Checkpoints
Download `checkpoints` from: https://uofi.box.com/s/wnt6cv7icuir4q3wb2a6viuyklme5dga. Put the checkpoints directories in `checkpoints` under `output/p3_finetuning`

# Setup Docker Environment
Run the following commands to initiate a docker container for running the scripts in this repo.
- `bash docker_build.sh`
- `bash run_container.sh`

# Quick Start
Scripts for performing (semi-)parametric multitask prompted training and zero-shot evaluation.

## No Aug baseline 

## Concat baseline

## FiD baseline

## Zemi


# Code Description
- code for the model architecture: `zemi/modeling_t5.py`, `zemi/modeling_xattn.py`
- code for multitask training: 
    - train No Aug and Concat baseline: `zemi/multi_task_fine_tune_baseline.py`
    - train FiD baseline and Zemi: `zemi/multi_task_fine_tune_xattn.py`
- code for zero-shot evaluation: 
    - eval No Aug and Concat baseline: `zemi/eval_original_task_only.py`
    - eval FiD baseline and Zemi: `zemi/eval_original_task_only_xattn.py`

# Citation
```
```
