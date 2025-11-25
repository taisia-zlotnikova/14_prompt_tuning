from all_models import my_model, peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import os
import json
from load_data import get_superglue_task
from train import make_train
from validate import validate_model

name_model = "t5-small"

def run_single_model_experiment(model_func, model_name, tokenizer, train_loader, val_loader, config, prompt_length=20):
    """Запускает эксперимент для одной модели"""
    print(f"   Testing {model_name}...")
    
    try:
        prompt_model = model_func(tokenizer, prompt_length=prompt_length, init_prompt=config['init_prompt'])
        train_loss = make_train(prompt_model, train_loader, lr=config['lr'])
        
        accuracy = validate_model(prompt_model, val_loader, tokenizer)
        print(f"   {model_name} Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"   ❌ {model_name} failed: {e}")
        return 0.0

def run_single_config_experiment(task_name, max_sizes, config, experiment_id=1):
    """Запускает один эксперимент с заданной конфигурацией для обеих моделей"""
    print(f"\n🧪 Experiment {experiment_id}")
    print(f"   LR: {config['lr']}, Balance: {config['balance_classes']}, Prompt: '{config['init_prompt']}'")
    
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    train_loader, val_loader, _ = get_superglue_task(
        task_name, tokenizer, batch_size=2, 
        max_sizes=max_sizes, balance_classes=config['balance_classes']
    )
    
    # Тестируем обе модели отдельно
    my_model_acc = run_single_model_experiment(my_model, "My Model", tokenizer, train_loader, val_loader, config)
    peft_model_acc = run_single_model_experiment(peft_model, "PEFT Model", tokenizer, train_loader, val_loader, config)
    
    print(f"   🎯 Results - My Model: {my_model_acc:.4f}, PEFT Model: {peft_model_acc:.4f}")
    
    return {
        'my_model': my_model_acc,
        'peft_model': peft_model_acc
    }


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
    
    # Отдельные лучшие результаты для каждой модели
    best_my_model = {'accuracy': 0, 'config': None}
    best_peft_model = {'accuracy': 0, 'config': None}
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\n🎯 Config {i+1}/{len(configs)}")
        print(f"   LR: {config['lr']}, Balance: {config['balance_classes']}, Prompt: '{config['init_prompt']}'")
        
        try:
            results = run_single_config_experiment(task_name, max_sizes, config, i+1)
            
            # Сохраняем результаты
            result_entry = {
                'config': config.copy(),
                'my_model_accuracy': results['my_model'],
                'peft_model_accuracy': results['peft_model']
            }
            all_results.append(result_entry)
            
            # Обновляем лучшие результаты для каждой модели
            if results['my_model'] > best_my_model['accuracy']:
                best_my_model = {
                    'accuracy': results['my_model'], 
                    'config': config.copy()
                }
                print(f"   🏆 NEW BEST My Model! Accuracy: {results['my_model']:.4f}")
            
            if results['peft_model'] > best_peft_model['accuracy']:
                best_peft_model = {
                    'accuracy': results['peft_model'],
                    'config': config.copy()
                }
                print(f"   🏆 NEW BEST PEFT Model! Accuracy: {results['peft_model']:.4f}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_results.append({
                'config': config.copy(),
                'my_model_accuracy': 0.0,
                'peft_model_accuracy': 0.0,
                'error': str(e)
            })
    
    # Сохраняем результаты
    save_results(task_name, all_results, best_my_model, best_peft_model, search_mode)
    
    return best_my_model, best_peft_model, all_results

def save_results(task_name, all_results, best_my_model, best_peft_model, search_mode):
    """Сохраняет результаты экспериментов"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{task_name}_{search_mode}_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump({
            'task': task_name,
            'search_mode': search_mode,
            'best_my_model': best_my_model,
            'best_peft_model': best_peft_model,
            'all_results': all_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"💾 Results saved to {filename}")

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
            
            # Получаем лучшие результаты для каждой модели
            best_my = data.get('best_my_model', {'accuracy': 0, 'config': {}})
            best_peft = data.get('best_peft_model', {'accuracy': 0, 'config': {}})
            
            # Сохраняем лучший результат для каждой задачи и каждой модели
            if task not in best_results:
                best_results[task] = {
                    'my_model': best_my,
                    'peft_model': best_peft,
                    'file': filename
                }
            else:
                # Обновляем если нашли лучше
                if best_my['accuracy'] > best_results[task]['my_model']['accuracy']:
                    best_results[task]['my_model'] = best_my
                if best_peft['accuracy'] > best_results[task]['peft_model']['accuracy']:
                    best_results[task]['peft_model'] = best_peft
    
    print("\n🏆 BEST RESULTS SUMMARY:")
    print("="*70)
    for task, results in best_results.items():
        print(f"📋 {task.upper():<10}")
        print(f"   My Model:  {results['my_model']['accuracy']:.4f}")
        print(f"     LR: {results['my_model']['config'].get('lr', 'N/A')}, "
              f"Balance: {results['my_model']['config'].get('balance_classes', 'N/A')}, "
              f"Prompt: '{results['my_model']['config'].get('init_prompt', 'N/A')}'")
        
        print(f"   PEFT Model: {results['peft_model']['accuracy']:.4f}")
        print(f"     LR: {results['peft_model']['config'].get('lr', 'N/A')}, "
              f"Balance: {results['peft_model']['config'].get('balance_classes', 'N/A')}, "
              f"Prompt: '{results['peft_model']['config'].get('init_prompt', 'N/A')}'")
        print(f"   File: {results['file']}")
        print("-" * 70)

def run_comprehensive_experiments():
    """Запускает все эксперименты с разными стратегиями"""
    # task_names = ['cb', 'boolq', 'copa', 'rte', 'wic']
    # task_names = ['copa', 'rte', 'wic']

    task_names = ['rte']
    max_sizes = {"train": None, "validation": None, "test": 1}
    
    # Тестируем разные стратегии
    strategies = ["smart"]
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"🚀 Running {strategy.upper()} strategy")
        print(f"{'='*70}")
        
        for task_name in task_names:
            print(f"\n📋 Task: {task_name}")
            best_my, best_peft, all_results = run_hyperparameter_search(
                task_name, max_sizes, search_mode=strategy
            )
            
            print(f"✅ {task_name} - Best My Model: {best_my['accuracy']:.4f}, Best PEFT: {best_peft['accuracy']:.4f}")

def run_individual_task(task_name, max_sizes, lr=0.01, init_prompt='', balance_classes=False, prompt_length=20, return_results=True):
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
                model_func, model_name, tokenizer, train_loader, val_loader, config, prompt_length
            )
            print(f"🎯 Final {model_name} Accuracy: {accuracy:.4f}")
            results[f'{model_name.lower().replace(" ", "_")}_accuracy'] = accuracy
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[f'{model_name.lower().replace(" ", "_")}_accuracy'] = 0.0
    
    print("🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺\n")
    if return_results:
        return results

def param_search():
    """Главная функция с разными режимами работы"""
    task_names = ['cb', 'boolq', 'copa', 'rte', 'wic']
    
    # Режим 1: Поиск гиперпараметров (рекомендуется)
    print("🚀 Starting hyperparameter search...")
    run_comprehensive_experiments()
    show_best_results_summary()
    
    # run_individual_task(
    #     task_name='cb',
    #     max_sizes={"train": None, "validation": None, "test": 1},
    #     lr=0.07,
    #     init_prompt='Classify the relationship:',
    #     balance_classes=True
    # )

    # Режим 2: Тест одной задачи с конкретными параметрами
    # for task_name in task_names:
    #     for lr in [0.0001, 0.001, 0.01, 0.03, 0.07, 0.1, 0.3]:
    #         run_individual_task(
    #             task_name=task_names,
    #             max_sizes={"train": None, "validation": None, "test": 1},
    #             lr=lr,
    #             init_prompt='Classify the relationship:',
    #             balance_classes=True
    #         )
    
    # Режим 3: Поиск для одной задачи
    # best_my, best_peft, _ = run_hyperparameter_search(
    #     task_name='cb',
    #     max_sizes={"train": 100, "validation": 100, "test": 1},
    #     search_mode="quick"
    # )
    # print(f"Best My Model: {best_my['accuracy']:.4f}")
    # print(f"Best PEFT Model: {best_peft['accuracy']:.4f}")