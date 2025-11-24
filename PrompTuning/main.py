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

# DEVICE
torch.cuda.set_device(4)

name_model = "t5-small"

def my_model(tokenizer, prompt_length=20, init_prompt=''):
    # Load T5
    model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
        
    prompt_model = PromptTuningWrapper(
        model,
        soft_prompt_length=prompt_length,
        hidden_dim=model.config.hidden_size,
        tokenizer=tokenizer,
        init_prompt = init_prompt
    ).to(device)

    return prompt_model

def peft_model(tokenizer, prompt_length=20, init_prompt=''):
    model = T5ForConditionalGeneration.from_pretrained(name_model)

    config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=prompt_length,
        prompt_tuning_init_text = init_prompt
    )
    prompt_model = get_peft_model(model, config).to(device)
    return prompt_model

def my_solve(task_name, model_func, max_sizes, lr, tokenizer, train_loader, val_loader, init_prompt):
    prompt_length = 20
    prompt_model = model_func(tokenizer, prompt_length, init_prompt)

    print("Training...")
    train_loss = make_train(prompt_model, train_loader, lr=lr)

    print("\nTrain:")
    train_metrics = validate_model(prompt_model, train_loader, tokenizer, desc="Train")
    print_metrics(train_metrics, phase="Train")

    print("\nValidation:")
    val_metrics = validate_model(prompt_model, val_loader, tokenizer, desc="Validation")
    print_metrics(val_metrics, phase="Validation")

def run_task(task_name, max_sizes, lr, init_prompt='', balance_classes=False):
    print("🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻")
    print(f"                   ▶️▶️▶️  RUNNING TASK {task_name}   ▶️▶️▶️ \n")

    tokenizer = AutoTokenizer.from_pretrained(name_model)
    # Load data
    train_loader, val_loader, test_loader = get_superglue_task(
        task_name,
        tokenizer,
        batch_size=2,
        max_sizes=max_sizes,
        balance_classes=balance_classes
    )
    print("\n🟢 Data loaded!")

    print("\n✏️✏️  My model:")
    my_solve(task_name, my_model, max_sizes=max_sizes, lr=lr, tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader, init_prompt=init_prompt)

    print("\n✏️✏️   Peft model:")
    my_solve(task_name, peft_model, max_sizes=max_sizes, lr=lr, tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader, init_prompt=init_prompt)

    print("🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺\n")
    print("\n➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖")
    print("➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖\n")

# run_task("cb", max_sizes={"train": None, "validation": None, "test": 1}, lr=0.003)
# run_task('boolq', max_sizes = {"train": None, "validation": None, "test": 1}, lr=0.001) # 0.001 
# run_task("copa", max_sizes={"train": None, "validation": None, "test": 1}, lr=0.003)
# run_task("rte", max_sizes={"train": None, "validation": None, "test": 1}, lr=0.003)


# run_task("cb", max_sizes={"train": 500, "validation": 500, "test": 1}, lr=0.03, init_prompt='', balance_classes=False)
# run_task('boolq', max_sizes = {"train": 500, "validation": 500, "test": 1}, lr=0.01, init_prompt='', balance_classes=False) # 0.001 
# run_task("copa", max_sizes={"train": 500, "validation": 500, "test": 1}, lr=0.03, init_prompt='', balance_classes=False)
# run_task("rte", max_sizes={"train": 500, "validation": 500, "test": 1}, lr=0.03, init_prompt='', balance_classes=False)
# run_task("wic", max_sizes={"train": 50, "validation": 50, "test": 1}, lr=0.03, init_prompt='', balance_classes=False)

task_names = ['cb', 'boolq', 'copa', 'rte', 'wic']
for task_name in task_names:
    max_sizes = {"train": 100, "validation": 100, "test": 1}

    run_task(task_name, max_sizes, lr=0.03, init_prompt='', balance_classes=False)