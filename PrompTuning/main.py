# main.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PromptTuningWrapper import PromptTuningWrapper
from load_data import get_superglue_task
from train import make_train
from validate import validate_model, print_metrics
from config import device

from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# Load T5
model_name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

s = "</s>"
s2 = "<pad>"
s3 = "."
print(tokenizer(s), tokenizer(s2), tokenizer(s3))

from peft import PromptTuningConfig, get_peft_model
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(model_name)

config = PromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=40
)

prompt_model = get_peft_model(model, config).to(device)

# Load data
train_loader, val_loader, test_loader = get_superglue_task(
    "boolq", tokenizer,
    batch_size=2,
    max_sizes={"train": 1000, "validation": 1000, "test": 1000}
)

print("Data loaded!")

print("Training...")
train_loss = make_train(prompt_model, train_loader, lr=0.03)
print("Train loss:", train_loss)

print("\nTrain:")
train_metrics = validate_model(prompt_model, train_loader, tokenizer)
print_metrics(train_metrics)

print("\nValidation:")
val_metrics = validate_model(prompt_model, val_loader, tokenizer)
print_metrics(val_metrics)
