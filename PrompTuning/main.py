from train import make_train
from train2 import make_train2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PromptTuningWrapper import PromptTuningWrapper
from load_data import get_superglue_task, tokenize_element
from validate import validate_model, print_metrics
from config import device

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
for p in model.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Загружаем датасет
max_sizes = {"train": 1000, "validation": 1000, "test": 100}
train_loader, val_loader, test_loader = get_superglue_task("boolq", tokenizer, max_sizes=max_sizes)

print("\nДанные успешно загружены!")
# -------------------------------------------------------------
prompt_lengths = [20]
test_accuracies = []

for prompt_length in prompt_lengths:
    print(f"Тестируем длину промпта: {prompt_length}")
    
    # Создаём модель
    prompt_model = PromptTuningWrapper(model, soft_prompt_length = prompt_length, hidden_dim=model.config.hidden_size).to(device)
    prompt_model.model.config.label_smoothing_factor = 0.1
    # Training
    # make_train2(prompt_model, train_loader)
    make_train(prompt_model, train_loader, lr = 0.5)
    print("\nОбучение завершено!")
    
    train_metrics = validate_model(prompt_model, train_loader, tokenizer, desc="Train")
    print_metrics(train_metrics, phase='Train')

    val_metrics = validate_model(prompt_model, val_loader, tokenizer, desc="Validation")
    print_metrics(val_metrics)


# ------------------------------------------------------------

# prompt_lengths = [15]
# test_accuracies = []

# for prompt_length in prompt_lengths:
#     print(f"Тестируем длину промпта: {prompt_length}")
    
#     # Создаём модель
#     prompt_model = PromptTuningWrapper(model, soft_prompt_length = prompt_length, hidden_dim=model.config.hidden_size).to(device)

#     # Оптимизатор
#     optimizer = torch.optim.Adam([prompt_model.soft_prompt], lr = 1e-3)

#     # Training
#     make_train(prompt_model, train_loader, optimizer)
#     print("\nОбучение завершено!")

#     train_metrics = validate_model(prompt_model, train_loader, tokenizer, desc="Train")
#     print_metrics(train_metrics, phase='Train')

#     val_metrics = validate_model(prompt_model, val_loader, tokenizer, desc="Validation")
#     print_metrics(val_metrics)

#     test_accuracies.append(val_metrics.get('Accuracy', 0))
#     print(f"Длина {prompt_length}: Val Accuracy = {test_accuracies[-1]:.4f}")
