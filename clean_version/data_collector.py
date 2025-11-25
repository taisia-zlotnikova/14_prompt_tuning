# data_collector.py

import json
import os
from datetime import datetime
from param_search import run_individual_task
from best_params import best_params

name_model = os.getenv('NAME_MODEL', 't5-small')
data_file_name = f"prompt_length_data_{name_model}.json"

def collect_prompt_length_data(name_model = 't5-small'):
    """Собирает данные по зависимости accuracy от длины промпта"""
    
    best_params_dict = best_params()
    task_names = list(best_params_dict.keys())
    prompt_lengths = [1, 5, 10, 20, 50, 100]
    
    # Файл для хранения данных
    data_file = data_file_name
    
    # Загружаем существующие данные или создаем новые
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = {}
    
    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"📊 Collecting data for: {task_name}")
        print(f"{'='*60}")
        
        params = best_params_dict[task_name]
        
        # Инициализируем данные для задачи если их нет
        if task_name not in all_data:
            all_data[task_name] = {
                'params': params,
                'experiments': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        
        for prompt_length in prompt_lengths:
            print(f"\n🔍 Testing prompt_length = {prompt_length}")
            
            # Проверяем, не тестировали ли уже эту конфигурацию
            existing_experiment = None
            for exp in all_data[task_name]['experiments']:
                if (exp['prompt_length'] == prompt_length and 
                    exp['lr'] == params['lr'] and
                    exp['init_prompt'] == params['init_prompt'] and
                    exp['balance_classes'] == params['balance_classes']):
                    existing_experiment = exp
                    break
            
            if existing_experiment:
                print(f"   ⏩ Already exists: My={existing_experiment['my_accuracy']:.4f}, PEFT={existing_experiment['peft_accuracy']:.4f}")
                continue
            
            # Запускаем эксперимент
            try:
                results = run_individual_task(
                    task_name=task_name,
                    max_sizes={"train": None, "validation": None, "test": 1},
                    lr=params['lr'],
                    init_prompt=params['init_prompt'],
                    balance_classes=params['balance_classes'],
                    prompt_length=prompt_length,
                    return_results=True,
                    name_model=name_model
                )
                
                # Сохраняем результаты
                experiment_data = {
                    'prompt_length': prompt_length,
                    'lr': params['lr'],
                    'init_prompt': params['init_prompt'],
                    'balance_classes': params['balance_classes'],
                    'my_accuracy': results.get('my_model_accuracy', 0),
                    'peft_accuracy': results.get('peft_model_accuracy', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
                all_data[task_name]['experiments'].append(experiment_data)
                all_data[task_name]['updated_at'] = datetime.now().isoformat()
                
                # Сохраняем после каждого эксперимента
                with open(data_file, 'w') as f:
                    json.dump(all_data, f, indent=2)
                
                print(f"   💾 Saved: My={experiment_data['my_accuracy']:.4f}, PEFT={experiment_data['peft_accuracy']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Experiment failed: {e}")
                continue
    
    print(f"\n✅ Data collection completed! Data saved to {data_file}")
    return all_data

def show_data_summary():
    """Показывает сводку собранных данных"""
    data_file = data_file_name
    
    if not os.path.exists(data_file):
        print("❌ No data file found. Run data collection first.")
        return
    
    with open(data_file, 'r') as f:
        all_data = json.load(f)
    
    print(f"\n📊 DATA SUMMARY ({len(all_data)} tasks):")
    print("="*50)
    
    for task_name, task_data in all_data.items():
        experiments = task_data['experiments']
        print(f"\n{task_name.upper():<10}: {len(experiments)} experiments")
        print(f"  Params: LR={task_data['params']['lr']}, Prompt='{task_data['params']['init_prompt']}'")
        
        if experiments:
            my_accuracies = [exp['my_accuracy'] for exp in experiments]
            peft_accuracies = [exp['peft_accuracy'] for exp in experiments]
            
            best_my = max(my_accuracies)
            best_peft = max(peft_accuracies)
            
            print(f"  Best My: {best_my:.4f}, Best PEFT: {best_peft:.4f}")
            print(f"  Updated: {task_data['updated_at'][:16]}")

if __name__ == "__main__":
    # Собираем данные
    data = collect_prompt_length_data('t5-small')
    
    # Показываем сводку
    show_data_summary()