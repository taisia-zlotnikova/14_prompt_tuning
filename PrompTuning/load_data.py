# load_data.py

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from collections import Counter
import random
import math

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

    target_enc = tokenizer(
        target_text,
        truncation=True,
        max_length=8,
        padding="max_length",
        return_tensors="pt"
    )
    labels = target_enc["input_ids"].squeeze(0)
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

def balance_dataset_with_max_size(dataset, task_name, max_size):
    """Балансирует датасет, беря из каждого класса max_size/cnt примеров"""
    # Группируем данные по меткам
    label_groups = {}
    for elem in dataset:
        label = elem["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(elem)
    
    cnt = len(label_groups)  # количество уникальных меток
    examples_per_class = max_size // cnt  # сколько брать из каждого класса
    
    print(f"Total classes: {cnt}, examples per class: {examples_per_class}")
    
    # Берем examples_per_class из каждого класса
    balanced_data = []
    for label, group in label_groups.items():
        if len(group) >= examples_per_class:
            # Берем ровно examples_per_class примеров
            balanced_data.extend(random.sample(group, examples_per_class))
            print(f"Class {label}: took {examples_per_class} from {len(group)}")
        else:
            # Если в классе меньше примеров, берем все что есть
            balanced_data.extend(group)
            print(f"Class {label}: took all {len(group)} (less than {examples_per_class})")
    
    # Перемешиваем данные
    random.shuffle(balanced_data)
    
    return balanced_data

def get_superglue_task(task_name, tokenizer, batch_size=2, max_sizes=None, balance_classes=True):

    dataset = load_dataset("super_glue", task_name)

    # БАЛАНСИРОВКА с учетом max_sizes
    if balance_classes and "train" in dataset:
        if max_sizes and "train" in max_sizes and max_sizes["train"]:
            max_train_size = max_sizes["train"]
            balanced_train_data = balance_dataset_with_max_size(dataset["train"], task_name, max_train_size)
            dataset["train"] = dataset["train"].from_list(balanced_train_data)
        else:
            # Если max_sizes не указан, используем обычную балансировку
            balanced_train_data = balance_dataset_with_max_size(dataset["train"], task_name, len(dataset["train"]))
            dataset["train"] = dataset["train"].from_list(balanced_train_data)
    
    # Применяем max_sizes к другим сплитам (без балансировки)
    if max_sizes:
        for split in ["validation", "test"]:
            if split in max_sizes and max_sizes[split] and split in dataset:
                dataset[split] = (
                    dataset[split].shuffle(seed=42).select(range(max_sizes[split]))
                )

    print("Final label distribution TRAIN:", Counter(dataset["train"]["label"]))

    # Токенизация после балансировки
    train_dataset = MyDataset(dataset["train"], tokenizer, task_name=task_name)
    val_dataset = MyDataset(dataset["validation"], tokenizer, task_name=task_name) if "validation" in dataset else None
    test_dataset = MyDataset(dataset["test"], tokenizer, task_name=task_name) if "test" in dataset else None

    loaders = [
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None,
        DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
    ]
    
    return [loader for loader in loaders if loader is not None]