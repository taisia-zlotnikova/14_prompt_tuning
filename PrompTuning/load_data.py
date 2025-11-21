
# load_data.py

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from collections import Counter


def tokenize_element(element, tokenizer, max_length=128):
    enc = tokenizer(
        element["passage"],
        element["question"],
        truncation=True,
        padding="max_length",   # or "longest"
        max_length=256,
    )
    enc["labels"] = torch.full((max_length,), -100)
    label = " yes" if element["label"] else " no"
    enc["labels"][-1] = torch.tensor(tokenizer([label])["input_ids"][0][0], dtype=torch.long)
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            enc[k] = v.clone().detach() 
        else:
            enc[k] = torch.tensor(v) 
    return {"input_ids" : enc["input_ids"],
            "attention_mask" : enc["attention_mask"],
            "labels" : enc["labels"]}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = [tokenize_element(elem, tokenizer) for elem in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_superglue_task(task_name, tokenizer, batch_size=2, max_sizes={"train": None, "validation": None, "test": None}):
    dataset = load_dataset("super_glue", task_name)

    for name, size in max_sizes.items():
        if size is not None:
            dataset[name] = dataset[name].shuffle(seed=42).select(range(size))

    print(Counter(dataset["train"]["label"][:]))

    train_dataset = MyDataset(dataset["train"], tokenizer)
    val_dataset = MyDataset(dataset["validation"], tokenizer)
    test_dataset = MyDataset(dataset["test"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader