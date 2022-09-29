# Data

## Download Preprocessed Datasets
- Download `dataset` from: https://uofi.box.com/s/wnt6cv7icuir4q3wb2a6viuyklme5dga
- Unzip the `30aug.zip`, put the folder under `data/p3_c4_document_level_chosen_examples`

## Using Custom Datasets & C4 Retrieval
Please refer to the README.md in [c4_retrieval/](../C4_retrieval/README.md)

## Data Format
The downloaded or processed datasets will be located at `data/p3_c4_document_level_chosen_examples/30aug`, which contains huggingface dataset instances that can be loaded with [load_from_disk](https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_from_disk). The dataset has the following fields:

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