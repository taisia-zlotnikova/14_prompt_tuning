# main.py

import torch
import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PromptTuningWrapper import PromptTuningWrapper
from load_data import get_superglue_task
from train import make_train
from validate import validate_model, print_metrics
from peft import PromptTuningConfig, get_peft_model
from transformers import T5ForConditionalGeneration
from config import device

# DEVICE
torch.cuda.set_device(4)
torch.cuda.empty_cache()

name_model = "t5-small"

def my_model(tokenizer, prompt_length=20, init_prompt=''):
    """Кастомная модель с PromptTuningWrapper"""
    model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
        
    prompt_model = PromptTuningWrapper(
        model,
        soft_prompt_length=prompt_length,
        hidden_dim=model.config.hidden_size,
        tokenizer=tokenizer,
        init_prompt=init_prompt
    ).to(device)

    return prompt_model

def peft_model(tokenizer, prompt_length=20, init_prompt=''):
    """PEFT модель"""
    model = T5ForConditionalGeneration.from_pretrained(name_model)
    
    config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=prompt_length,
        prompt_tuning_init_text=init_prompt
    )
    prompt_model = get_peft_model(model, config).to(device)
    return prompt_model

def run_single_experiment(task_name, max_sizes, config, experiment_id=1):
    """Запускает один эксперимент с заданной конфигурацией"""
    print(f"\n🧪 Experiment {experiment_id}")
    print(f"   LR: {config['lr']}, Balance: {config['balance_classes']}, Prompt: '{config['init_prompt']}'")
    
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    train_loader, val_loader, _ = get_superglue_task(
        task_name, tokenizer, batch_size=2, 
        max_sizes=max_sizes, balance_classes=config['balance_classes']
    )
    
    # Тестируем обе модели
    accuracies = []
    
    for model_name, model_func in [("My Model", my_model), ("PEFT Model", peft_model)]:
        print(f"\n   Testing {model_name}...")
        
        try:
            prompt_model = model_func(tokenizer, prompt_length=20, init_prompt=config['init_prompt'])
            train_loss = make_train(prompt_model, train_loader, lr=config['lr'])
            
            val_accuracy = validate_model(prompt_model, val_loader, tokenizer)
            accuracies.append(val_accuracy)
            
            print(f"   {model_name} Accuracy: {val_accuracy:.4f}")
        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")
            accuracies.append(0.0)
    
    # Возвращаем среднюю точность
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"   🎯 Average Accuracy: {avg_accuracy:.4f}")
    
    return avg_accuracy

def run_hyperparameter_search(task_name, max_sizes, search_mode="smart"):
    """Запускает поиск гиперпараметров для задачи"""
    from hyperparameters import get_hyperparameter_grid, get_quick_grid, PRESET_CONFIGS
    
    # Выбираем стратегию поиска
    if search_mode == "quick":
        configs = get_quick_grid(task_name)
    elif search_mode == "preset":
        configs = PRESET_CONFIGS.get(task_name, get_quick_grid(task_name))
    else:  # "smart"
        configs = get_hyperparameter_grid(task_name)
    
    print(f"🔍 Testing {len(configs)} configurations for {task_name} ({search_mode} mode)")
    
    best_accuracy = 0
    best_config = None
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n🎯 Config {i+1}/{len(configs)}")
        print(f"   LR: {config['lr']}, Balance: {config['balance_classes']}, Prompt: '{config['init_prompt']}'")
        
        try:
            accuracy = run_single_experiment(task_name, max_sizes, config, i+1)
            
            results.append({
                'config': config.copy(),
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config.copy()
                print(f"   🏆 NEW BEST! Accuracy: {accuracy:.4f}")
            else:
                print(f"   📊 Accuracy: {accuracy:.4f}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'config': config.copy(),
                'accuracy': 0.0,
                'error': str(e)
            })
    
    # Сохраняем результаты
    save_results(task_name, results, best_config, search_mode, best_accuracy)
    
    return best_config, best_accuracy, results

def save_results(task_name, results, best_config, search_mode, best_accuracy):
    """Сохраняет результаты экспериментов"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{task_name}_{search_mode}_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    
    # Добавляем accuracy в best_config
    best_config_with_accuracy = best_config.copy()
    best_config_with_accuracy['accuracy'] = best_accuracy
    
    with open(filename, 'w') as f:
        json.dump({
            'task': task_name,
            'search_mode': search_mode,
            'best_config': best_config_with_accuracy,  # ← с accuracy внутри
            'best_accuracy': best_accuracy,           # ← и отдельно
            'all_results': results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"💾 Results saved to {filename}")


def run_individual_task(task_name, max_sizes, lr=0.01, init_prompt='', balance_classes=False):
    """Запускает одну задачу с конкретными параметрами (старый формат)"""
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
    for model_name, model_func in [("My Model", my_model), ("PEFT Model", peft_model)]:
        print(f"\n✏️  {model_name}:")
        
        try:
            prompt_model = model_func(tokenizer, prompt_length=20, init_prompt=init_prompt)
            
            print("Training...")
            train_loss = make_train(prompt_model, train_loader, lr=lr)
            print(f"Train loss: {train_loss:.4f}")

            print("\nValidation:")
            val_metrics = validate_model(prompt_model, val_loader, tokenizer, desc="Validation")
            print_metrics(val_metrics, phase="Validation")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    print("🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺\n")

def show_best_results_summary():
    """Показывает сводку лучших результатов по всем задачам"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ No results found")
        return
    
    best_results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            task = data['task']
            accuracy = data['best_config'].get('accuracy', 0)
            
            # Сохраняем лучший результат для каждой задачи
            if task not in best_results or accuracy > best_results[task]['accuracy']:
                best_results[task] = {
                    'accuracy': accuracy,
                    'config': data['best_config'],
                    'file': filename
                }
    
    print("\n🏆 BEST RESULTS SUMMARY:")
    print("="*60)
    for task, result in best_results.items():
        print(f"📋 {task.upper():<10} {result['accuracy']:.4f}")
        print(f"   LR: {result['config']['lr']}, "
              f"Balance: {result['config']['balance_classes']}, "
              f"Prompt: '{result['config']['init_prompt']}'")
        print(f"   File: {result['file']}")
        print("-" * 40)


def run_comprehensive_experiments():
    """Запускает все эксперименты с разными стратегиями"""
    task_names = ['cb', 'boolq', 'copa', 'rte', 'wic']
    # task_names = ['cb']
    max_sizes = {"train": None, "validation": None, "test": 1}
    
    # Тестируем разные стратегии
    strategies = ["quick", "preset"]  # "smart" можно добавить позже
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"🚀 Running {strategy.upper()} strategy")
        print(f"{'='*60}")
        
        for task_name in task_names:
            print(f"\n📋 Task: {task_name}")
            best_config, best_acc, results = run_hyperparameter_search(
                task_name, max_sizes, search_mode=strategy
            )
            
            print(f"✅ {task_name} - Best: {best_acc:.4f} with {best_config}")

def main():
    """Главная функция с разными режимами работы"""
    
    # Режим 1: Быстрый поиск гиперпараметров (рекомендуется)
    print("🚀 Starting hyperparameter search...")
    run_comprehensive_experiments()
    show_best_results_summary()
    
    # Режим 2: Тест одной задачи с конкретными параметрами
    # run_individual_task(
    #     task_name='boolq',
    #     max_sizes={"train": 100, "validation": 100, "test": 1},
    #     lr=0.01,
    #     init_prompt='Answer the question:',
    #     balance_classes=True
    # )
    
    # Режим 3: Поиск для одной задачи
    # run_hyperparameter_search(
    #     task_name='cb',
    #     max_sizes={"train": 100, "validation": 100, "test": 1},
    #     search_mode="quick"
    # )

if __name__ == "__main__":
    main()