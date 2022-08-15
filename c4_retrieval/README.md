# C4 Retrieval

## Setup ElasticSearch Client
- Set up ElasticSearch following https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html
- Set up `ELASTIC_PASSWORD` and `PATH_TO_HTTP_CA_CRT` in `es_client.py`
- Download C4 dataset from Huggingface Datasets: https://huggingface.co/datasets/c4
- Set up `c4_local_path` in `index.py`
- Run `python index.py` to generate index for C4 dataset (default using 5%)

## Perform C4 Retrieval on Huggingface Datasets
- Specify a list of datasets for retrieval in `tasks/tasks_to_be_processed.json`
- Run `python _run_p3_C4_document_level_retrieval.py`
- The output datasets will be stored at `/data/p3_c4_document_level`

## Post Processing
script for post process the retrieved datasets into our input format.
- specify the `TASK` variable in `_run_process_p3_C4_retrieval_datasets.py` you want to process
- Run `python _run_process_p3_C4_retrieval_datasets`
