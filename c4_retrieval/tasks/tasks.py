import random
from datasets import load_dataset
from promptsource.templates import DatasetTemplates




if __name__ == "__main__":
    random.seed(11)
    task_lines = open("task_list.txt", 'r').readlines()
    task_names = []
    for line in task_lines:
        if not line.strip():
            continue
        if line[0] == "#":
            continue
        task_name = line.strip()
        if task_name not in task_names:
            task_names.append(task_name)

    output = open("output.txt", 'w', 1)
    print("len(task_names):", len(task_names))

    #start_idx = task_names.index("cos_e/v1.11")

    for i, task_name in enumerate(task_names):
        print(i, task_name)
        task_name = task_name.strip()


        #if i < start_idx:
        #	continue

        try:
            if task_name == "anli":
                dataset = load_dataset(task_name, split="train_r1", cache_dir = '/cephfs/KERR_data/huggingface_datasets/')
            elif "/" in task_name:
                dataset_name = task_name.split("/")[0]
                subset_name = task_name.split("/")[1]
                dataset = load_dataset(dataset_name, subset_name, split="train", cache_dir = '/cephfs/KERR_data/huggingface_datasets/')
            else:
                dataset_name = task_name
                dataset = load_dataset(dataset_name, split="train", cache_dir = '/cephfs/KERR_data/huggingface_datasets/')
        except FileNotFoundError:
            print("FileNotFoundError: " + task_name)
            output.write("\nFileNotFoundError: " + task_name + "\n\n")
            continue

        output.write("########## " + task_name + "\n")
        output.write("$$$$$$$$$$ " + " ".join(list(dataset[0].keys())) + "\n")
        for instance in list(dataset)[:5]:
            output.write(str(instance) + "\n")
        output.write("\n\n")
    output.close()




    