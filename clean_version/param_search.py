from all_models import my_model, peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import os
import json
from load_data import get_superglue_task
from train import make_train
from validate import validate_model

def run_single_model_experiment(model_func, model_name, tokenizer, train_loader, val_loader, config, prompt_length=20, name_model="t5-small"):
    """Запускает эксперимент для одной модели"""
    print(f"   Testing {model_name}...")
    
    try:
        prompt_model = model_func(tokenizer, prompt_length=prompt_length, init_prompt=config['init_prompt'], name_model=name_model)
        train_loss = make_train(prompt_model, train_loader, lr=config['lr'])
        
        accuracy = validate_model(prompt_model, val_loader, tokenizer)
        print(f"   {model_name} Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"   ❌ {model_name} failed: {e}")
        return 0.0

def run_individual_task(task_name, max_sizes, lr=0.01, init_prompt='', balance_classes=False, prompt_length=20, return_results=True, name_model="t5-small"):
    """Запускает одну задачу с конкретными параметрами"""
    print(f"\n🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻")
    print(f"                   ▶️  RUNNING TASK {task_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(name_model)
    train_loader, val_loader, test_loader = get_superglue_task(
        task_name,
        tokenizer,
        batch_size=2,
        max_sizes=max_sizes,
        balance_classes=balance_classes
    )
    print("\n🟢 Data loaded!")

    # Тестируем обе модели последовательно
    config = {'lr': lr, 'init_prompt': init_prompt, 'balance_classes': balance_classes}
    print(f'\n🧩  config = {config}')

    results = {}
    
    for model_name, model_func in [("My Model", my_model), ("PEFT Model", peft_model)]:
        print(f"\n✏️  {model_name}:")
        
        try:
            accuracy = run_single_model_experiment(
                model_func, model_name, tokenizer, train_loader, val_loader, config, prompt_length, name_model=name_model
            )
            print(f"🎯 Final {model_name} Accuracy: {accuracy:.4f}")
            results[f'{model_name.lower().replace(" ", "_")}_accuracy'] = accuracy
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[f'{model_name.lower().replace(" ", "_")}_accuracy'] = 0.0
    
    print("🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺\n")
    if return_results:
        return results
