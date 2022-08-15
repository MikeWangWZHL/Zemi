import os
import argparse
import logging

import ujson as json
from datasets import load_dataset, load_from_disk

import es_client

from tqdm import tqdm

PWD = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


# read valid queries for each dataset
def load_datasets_metadata(file):
    dataset2query = {}
    input_lines = open(file, 'r')
    for line in input_lines:
        if not line.strip():
            continue
        if line[0] == "#":
            continue
        task_name = line.split(" | ")[0]
        query = line.split(" | ")[1].split()
        dataset2query[task_name] = query
    input_lines.close()
    return dataset2query

def load_c4_dataset(es_index_name = "hf_c4_text", portion = "all"):
    c4 = load_from_disk('/cephfs/user/xiaomanpan/data/c4/c4_en')
    if portion == "all":
        c4 = c4['train']
    elif portion == "5percent":
        c4 = c4['train'].select(range(18243444))

    es_config = {
        "settings": {
            "number_of_shards": 16,
            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                    "similarity": "BM25"
                },
            }
        },
    }
    # es_index_name = "hf_c4_text"
    # es_index_name = "hf_c4_text_5percent"
    c4.load_elasticsearch_index(
        "text", es_client=es_client.client, es_index_name=es_index_name, es_index_config=es_config)
    return c4

def get_query_string(q, max_query_token_length = 20):
    if isinstance(q, str):
        ret = q
    elif isinstance(q, list):
        ret = "\n".join(q)
    elif isinstance(q, dict):
        if "text" in q:
            ret = "\n".join(q['text'])
    else:
        raise ValueError
    # truncate length
    ret = ' '.join(ret.split(' ')[:max_query_token_length])    
    return ret

def get_query_string_batch(qs, max_query_token_length = 20):
    return [get_query_string(q, max_query_token_length) for q in qs]


def truncate_document(document, max_length=1024):
    # print(len(document.split(' ')))
    document = ' '.join(document.split(' ')[:max_length])
    return document


def retrieve(client, index_name, query_text, k=10):
    response = client.options(request_timeout=100).search(
        index = index_name, 
        query = {"multi_match": {"query": query_text, "fields": ["text"], "type": "cross_fields"}}, size=k
    )
    rets = response['hits']['hits']
    return rets


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
        "--task_list",
        type=str,
        default=None,
        help="a list of huggingface task names : 'openbookqa/main', 'piqa', etc",
    )
    # parser.add_argument(
    #     "--dataset_path",
    #     type=str,
    #     default=None,
    #     help="dataset path on disk to load from",
    # )
    # parser.add_argument(
    #     "--query_keys",
    #     type=str,
    #     nargs='+',
    #     default=[],
    #     help="query keys to do retrieval"
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory.",
        required=True,
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use.",
    )
    args = parser.parse_args()

    return args



# TASK_LIST = [
#     "cos_e/v1.11",
#     "cosmos_qa",
#     "dream",
#     "openbookqa/main",
#     "piqa",
#     "qasc",
#     "quartz",
#     "sciq",
#     "social_i_qa",
#     "wiqa"
# ]
# TASK_LIST = [
#     # "imdb",
#     "rotten_tomatoes",
#     "trec",
#     "super_glue/cb",
#     "super_glue/copa",
#     "anli",
#     "hellaswag",
#     "super_glue/wic",
#     "wiki_qa",
#     "hotpot_qa",
#     # "openbookqa/main",
#     # "piqa",
# ]
TASK_LIST=None

def main():
    args = parse_args()
    
    MAX_RETRIEVAL_NUM = 30
    MAX_QUERY_LENGTH = 20
    MAX_DOCUMENT_LENGTH = 256
    if_truncate_document = True

    es_index_name = "hf_c4_text_5percent"
    portion = "5percent"


    ## load c4 dataset
    logger.info("loading c4...")
    c4 = load_c4_dataset(es_index_name=es_index_name, portion=portion)
    logger.info("loaded c4!")
    logger.info(c4)

    if args.task_list is not None or TASK_LIST is not None:
        if TASK_LIST is not None:
            print("using task list defined in this script...")
            task_list = TASK_LIST
            print(task_list)
        else:
            print("loading task list from:", args.task_list)
            task_list = json.load(open(args.task_list))
        to_be_processed = []
        for t in task_list:
            if '/' in t:
                dataset_name, dataset_config_name = t.split("/")
            else:
                dataset_name, dataset_config_name = t, None
            to_be_processed.append((dataset_name, dataset_config_name))
    elif args.task_list is None and args.dataset_name is not None:
        to_be_processed = [(args.dataset_name, args.dataset_config_name)]
    else:
        raise ValueError('Please specify `args.task_list` or `args.dataset_name and args.dataset_config_name` as appear in `promptsource`.')
    
    logger.info("datasets to be processed:")
    logger.info(to_be_processed)

    ###
    output_dir = args.output_dir
    num_proc = args.num_proc

    for dataset_name, dataset_config_name in to_be_processed:
        
        if dataset_config_name is None:
            task_name = dataset_name
        else:
            task_name = dataset_name + "/" + dataset_config_name

        # set up output dir
        output_dataset_path = os.path.join(output_dir, task_name.replace("/","_"))
        if os.path.isdir(output_dataset_path) and os.stat(output_dataset_path).st_size > 0:
            print(f'nonempty dir exists, skip: {dataset_name} {dataset_config_name}')
            continue
        os.makedirs(output_dataset_path, exist_ok=True)
        
        # load dataset
        raw_dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            # download_mode="force_redownload"
            # cache_dir = '/cephfs/KERR_data/huggingface_datasets/'
        )
        logger.info(f"loaded_dataset: {dataset_name} {dataset_config_name}")
        logger.info(raw_dataset)

        # load query keys
        dataset2query = load_datasets_metadata(f"{PWD}/tasks/task_query_keys_c4_retrieval_minimum.txt")
        if dataset_config_name is None:
            query = dataset2query[dataset_name]
        else:
            query = dataset2query[dataset_name + "/" + dataset_config_name]
        logger.info("loaded query keys:")
        logger.info(query)

        # def c4_retrieval(example):
        #     retrieved_examples = {q:[] for q in query}
        #     try:
        #         for q in query:
        #             api_query_string = get_query_string(example[q])
        #             if api_query_string is None:
        #                 print("error key!!!!:", example[q])
        #                 continue
        #             # print(api_query_string)
        #             rets = retrieve(es_client.client, es_index_name, api_query_string, k=MAX_RETRIEVAL_NUM)
        #             # print(rets)
        #             # print('====================================')
        #             if if_truncate_document:
        #                 retrieved_examples[q] = [
        #                     {
        #                         "score": item["_score"],
        #                         "text": truncate_document(item["_source"]["text"], max_length=MAX_DOCUMENT_LENGTH)
        #                     } for item in rets
        #                 ]
        #             else:
        #                 retrieved_examples[q] = [
        #                     {
        #                         "score": item["_score"],
        #                         "text": item["_source"]["text"]
        #                     } for item in rets
        #                 ]
                    
        #             # for item in retrieved_examples[q]:
        #             #     print(item)
        #             # quit()
        #             # scores, rets = c4.get_nearest_examples("text", api_query_string, k=MAX_RETRIEVAL_NUM)
        #             # retrieved_examples[q] = [
        #             #     {
        #             #         "score": scores[i],
        #             #         "text": rets['text'][i]
        #             #     } for i in range(len(scores))
        #             # ]

        #     except Exception as e:
        #         logger.error('unexpected error for c4 search')
        #         logger.exception(e)
        #     example['retrieved_examples'] = retrieved_examples
        #     return example
        # # do retrieval
        # processed_dataset = raw_dataset.map(
        #     c4_retrieval,
        #     num_proc=num_proc
        # )
        # processed_dataset.save_to_disk(f'{output_dataset_path}')
        
        ## get nearest batch
        def c4_retrieval(examples):
            bs = len(examples[query[0]])
            retrieved_examples = [ {q:[] for q in query} for _ in range(bs) ]
            try:
                for q in query:
                    api_query_string_batch = get_query_string_batch(examples[q], max_query_token_length = MAX_QUERY_LENGTH)
                    scores, rets = c4.get_nearest_examples_batch("text", api_query_string_batch, k=MAX_RETRIEVAL_NUM)
                    assert bs == len(scores)
                    for i in range(bs):
                        ith_scores = scores[i]
                        ith_rets = rets[i]
                        retrieved_examples[i][q] = [
                            {
                                "score": ith_scores[j],
                                "text": truncate_document(ith_rets['text'][j], max_length=MAX_DOCUMENT_LENGTH)
                            } for j in range(len(ith_scores))
                        ]
            except Exception as e:
                logger.error('unexpected error for c4 search')
                logger.exception(e)
            examples['retrieved_examples'] = retrieved_examples
            return examples
        # do retrieval
        processed_dataset = raw_dataset.map(
            c4_retrieval,
            num_proc=num_proc,
            batched=True
        )
        processed_dataset.save_to_disk(f'{output_dataset_path}')


if __name__ == '__main__':
    main()