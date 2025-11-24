# main.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PromptTuningWrapper import PromptTuningWrapper
from load_data import get_superglue_task
from train import make_train
from validate import validate_model, print_metrics
from config import device

# Load T5
name_model = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(device)
tokenizer = AutoTokenizer.from_pretrained(name_model)

# Freeze base model
for p in model.parameters():
    p.requires_grad = False

# Load data
train_loader, val_loader, test_loader = get_superglue_task(
    "boolq", tokenizer,
    batch_size=2,
    max_sizes={"train": 500, "validation": 500, "test": 200}
)

print("Data loaded!")

prompt_length = 20
prompt_model = PromptTuningWrapper(
    model,
    soft_prompt_length=prompt_length,
    hidden_dim=model.config.hidden_size,
    tokenizer=tokenizer
).to(device)

print("Training...")
train_loss = make_train(prompt_model, train_loader, lr=0.0001)
print("Train loss:", train_loss)

print("\nTrain:")
train_metrics = validate_model(prompt_model, train_loader, tokenizer)
print_metrics(train_metrics)

print("\nValidation:")
val_metrics = validate_model(prompt_model, val_loader, tokenizer)
print_metrics(val_metrics)
