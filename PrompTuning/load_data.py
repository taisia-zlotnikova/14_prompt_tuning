# load_data.py

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from collections import Counter

def tokenize_element(element, tokenizer, max_length=256, target_max_len=4):
    # Encode encoder input
    enc = tokenizer(
        element["passage"],
        element["question"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # Encode decoder target
    target = "yes" if element["label"] else "no"
    t = tokenizer(
        target,
        truncation=True,
        padding="max_length",
        max_length=target_max_len
    )
    labels = torch.tensor(t["input_ids"], dtype=torch.long)

    # Convert encoder inputs to tensors
    for k in ["input_ids", "attention_mask"]:
        enc[k] = torch.tensor(enc[k], dtype=torch.long)

    enc["labels"] = labels

    return enc

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = [tokenize_element(elem, tokenizer) for elem in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_superglue_task(task_name, tokenizer, batch_size=2, max_sizes=None):

    dataset = load_dataset("super_glue", task_name)

    if max_sizes:
        for split in ["train", "validation", "test"]:
            if split in max_sizes and max_sizes[split]:
                dataset[split] = (
                    dataset[split].shuffle(seed=42).select(range(max_sizes[split]))
                )
    print("Label distribution TRAIN:", Counter(dataset["train"]["label"]))
    print("Label distribution VALIDATE:", Counter(dataset["validation"]["label"]))

    train_dataset = MyDataset(dataset["train"], tokenizer)
    val_dataset = MyDataset(dataset["validation"], tokenizer)
    test_dataset = MyDataset(dataset["test"], tokenizer)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )
