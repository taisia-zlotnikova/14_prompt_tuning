# load_data.py

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from collections import Counter

def tokenize_element(element, tokenizer, prompt_length=5, task_name='cb'):
    if task_name == 'boolq':
        input_text = f"question: {element['question']} context: {element['passage']}"
        target_text = "yes" if element["label"] else "no"
    elif task_name == 'cb':
        input_text = f"premise: {element['premise']} hypothesis: {element['hypothesis']}"
        label = element["label"]
        if label == 0:
            target_text = "entailment"
        elif label == 1:
            target_text = "contradiction"
        else:
            target_text = "neutral"

    # Важно: НЕ добавляем EOS в target вручную — T5 сам добавит при генерации
    target_enc = tokenizer(
        target_text,
        truncation=True,
        max_length=8,
        padding="max_length",
        return_tensors="pt"
    )
    labels = target_enc["input_ids"].squeeze(0)
    # Заменяем pad на -100 сразу (это хорошая практика)
    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)

    input_enc = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512 - prompt_length,
        return_tensors="pt"
    )

    return {
        "input_ids": input_enc["input_ids"].squeeze(0),
        "attention_mask": input_enc["attention_mask"].squeeze(0),
        "labels": labels
    }

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, task_name):
        self.data = [tokenize_element(elem, tokenizer, task_name=task_name) for elem in data]

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

    train_dataset = MyDataset(dataset["train"], tokenizer, task_name=task_name)
    val_dataset = MyDataset(dataset["validation"], tokenizer, task_name=task_name)
    test_dataset = MyDataset(dataset["test"], tokenizer, task_name=task_name)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )
