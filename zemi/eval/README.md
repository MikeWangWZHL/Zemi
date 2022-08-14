# Evaluation Jobs

The lowest level scripts that will be called are: (in case you need to set up some ENV at the beginning)
- `run_eval_finetuned_mixture_baseline.sh`
- `run_eval_finetuned_mixture_xattn.sh`

## Aug 3 jobs to be run:
- `run_eval_jobs_docker_8_3_0.sh`
- `run_eval_jobs_docker_8_3_1.sh`
- `run_eval_jobs_docker_8_3_2.sh`
- `run_eval_jobs_docker_8_3_3.sh`
- `run_eval_jobs_docker_8_3_4.sh`

Note: same as training jobs, each scripts above will call `../SETUP_DOCKER_ENV.sh` to set up some ENV, please double check if those are aligned with the cluster envirionment, thanks!