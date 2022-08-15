from datasets import load_from_disk

import es_client

print('loading c4 ...')
c4_local_path = '<path to download C4 dataset>'
c4 = load_from_disk(c4_local_path)
c4 = c4['train'].select(range(18243444)) # sample 5%

print(c4)

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
es_index_name = "hf_c4_text_5percent"  # name of the index in ElasticSearch
# es_client.client.indices.delete(index=es_index_name)
c4.add_elasticsearch_index(
    "text", es_client=es_client.client, es_index_config=es_config, es_index_name=es_index_name)


## perform search ##

# es_index_name = "hf_c4_text"
# c4.load_elasticsearch_index(
#     "text", es_client=es_client.client, es_index_name=es_index_name)
# query = "let's think step by step"
# scores, retrieved_examples = c4.get_nearest_examples("text", query, k=5)
# for i, j in zip(scores, retrieved_examples['text']):
#     print(i)
#     print(j)
#     print()