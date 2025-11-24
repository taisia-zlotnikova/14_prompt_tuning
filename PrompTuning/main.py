# main.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PromptTuningWrapper import PromptTuningWrapper
from load_data import get_superglue_task
from train import make_train
from validate import validate_model, print_metrics
from config import device
torch.cuda.empty_cache()

torch.cuda.set_device(4)

def solve(task_name, max_sizes, lr=0.1):
    # Load T5
    name_model = "t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name_model)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    # Load data
    train_loader, val_loader, test_loader = get_superglue_task(
        task_name,
        tokenizer,
        batch_size=2,
        max_sizes=max_sizes
    )
    print("Data loaded!")

    prompt_length = 10
    prompt_model = PromptTuningWrapper(
        model,
        soft_prompt_length=prompt_length,
        hidden_dim=model.config.hidden_size,
        tokenizer=tokenizer
    ).to(device)

    print("Training...")
    train_loss = make_train(prompt_model, train_loader, lr=lr)
    print("Train loss:", train_loss)

    print("\nTrain:")
    train_metrics = validate_model(prompt_model, train_loader, tokenizer)
    print_metrics(train_metrics)

    print("\nValidation:")
    val_metrics = validate_model(prompt_model, val_loader, tokenizer)
    print_metrics(val_metrics)

solve("cb", max_sizes = {"train": None, "validation": None, "test": 1}, lr=0.005)
# solve('boolq', max_sizes = {"train": 1000, "validation": 500, "test": 1}, lr=0.001)