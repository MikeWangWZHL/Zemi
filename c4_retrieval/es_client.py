from elasticsearch import Elasticsearch

ELASTIC_PASSWORD = "<your_password>"
PATH_TO_HTTP_CA_CRT = "<path_to_your_http_ca.crt>"

## Create the client instance
client = Elasticsearch(
    "https://localhost:9200",
    ca_certs=PATH_TO_HTTP_CA_CRT,
    basic_auth=("elastic", ELASTIC_PASSWORD),
    request_timeout=100
)

# Successful response!
print(client.info())
