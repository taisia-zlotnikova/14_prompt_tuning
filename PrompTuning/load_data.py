# load_data.py

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from collections import Counter
def tokenize_element(element, tokenizer, prompt_length=10):
    input_text = f"question: {element['question']} context: {element['passage']}"
    target_text = "yes" if element["label"] else "no"
    
    # Токенизируем вход
    input_enc = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=512 - prompt_length,
        return_tensors="pt"
    )
    
    # Токенизируем цель БЕЗ паддинга!
    target_enc = tokenizer(
        target_text,
        add_special_tokens=True,  # Для T5 обычно нужны специальные токены
        return_tensors="pt"       # НЕ используем padding="max_length"!
    )
    
    return {
        "input_ids": input_enc["input_ids"].squeeze(0),
        "attention_mask": input_enc["attention_mask"].squeeze(0),
        "labels": target_enc["input_ids"].squeeze(0)  # Теперь это [token_id, eos_token_id]
    }

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
