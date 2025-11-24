# main.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PromptTuningWrapper import PromptTuningWrapper
from load_data import get_superglue_task
from train import make_train
from validate import validate_model, print_metrics
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from transformers import T5ForConditionalGeneration
from config import device
torch.cuda.empty_cache()
torch.cuda.set_device(5)

name_model = "t5-small"

def my_model(tokenizer, prompt_length=10):
    # Load T5
    model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    prompt_model = PromptTuningWrapper(
        model,
        soft_prompt_length=prompt_length,
        hidden_dim=model.config.hidden_size,
        tokenizer=tokenizer
    ).to(device)

    return prompt_model

def peft_model(tokenizer, prompt_length=10):
    model = T5ForConditionalGeneration.from_pretrained(name_model)

    config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=prompt_length,
        prompt_tuning_init_text="Is the answer to given question true or false?"
    )
    prompt_model = get_peft_model(model, config).to(device)
    return prompt_model

def my_solve(task_name, model_func, max_sizes, lr, tokenizer, train_loader, val_loader):
    prompt_length = 10
    prompt_model = model_func(tokenizer, prompt_length)

    print("Training...")
    train_loss = make_train(prompt_model, train_loader, lr=lr)

    print("\nTrain:")
    train_metrics = validate_model(prompt_model, train_loader, tokenizer, desc="Train")
    print_metrics(train_metrics, phase="Train")

    print("\nValidation:")
    val_metrics = validate_model(prompt_model, val_loader, tokenizer, desc="Validation")
    print_metrics(val_metrics, phase="Validation")

def run_task(task_name, max_sizes, lr):
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    # Load data
    train_loader, val_loader, test_loader = get_superglue_task(
        task_name,
        tokenizer,
        batch_size=2,
        max_sizes=max_sizes
    )
    print("Data loaded!")

    my_solve(task_name, my_model, max_sizes=max_sizes, lr=lr, tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader)
    # my_solve(task_name, peft_model, max_sizes=max_sizes, lr=lr, tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader)

# run_task("cb", max_sizes={"train": None, "validation": None, "test": 1}, lr=0.003)
run_task('boolq', max_sizes = {"train": 1000, "validation": 500, "test": 1}, lr=0.0001)