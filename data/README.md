# Data

## Download Preprocessed Datasets
TODO: ADD LINK

## Using Custom Datasets
Please refer to [C4 Retrieval and Post Processing Instruction](../C4_retrieval/README.md)

## Data Format
The downloaded or processed datasets will be in `data/p3_c4_document_level_chosen_examples/30aug` which are huggingface dataset instances that can be loaded with [load_from_disk](https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_from_disk). The dataset has the following fields:

| Dataset fields | Description |
--- | --- |
| `inputs_pretokenized` | prompted input text (str)
| `targets_pretokenized` | target text (str)
| `answer_choices` | candidate answer choices (dict or list) 
| `chosen_examples` | list of retrieved augmentations

Only one field in each `chosen_examples` instance are actually used in our experiments (other fields are for future extension which can be ignored):
| `chosen_examples` instance fields | Description |
--- | --- |
| `inputs_pretokenized` | raw text of the retrieved augmentaiton