# main.py

import matplotlib.pyplot as plt
import numpy as np
from param_search import run_individual_task
from best_params import best_params

def run_prompt_length_analysis():
    """Запускает анализ зависимости accuracy от длины промпта и строит графики"""
    
    # Получаем лучшие параметры для каждой задачи
    best_params_dict = best_params()
    task_names = list(best_params_dict.keys())
    prompt_lengths = [5, 10, 15, 20, 25, 30]
    
    # Словари для хранения результатов
    all_results = {}
    
    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"📊 Analyzing prompt length for: {task_name}")
        print(f"{'='*60}")
        
        params = best_params_dict[task_name]
        task_results = {
            'my_model': [],
            'peft_model': [],
            'lengths': prompt_lengths
        }
        
        for prompt_length in prompt_lengths:
            print(f"\n🔍 Testing prompt_length = {prompt_length}")
            
            # Запускаем задачу с конкретной длиной промпта
            results = run_individual_task(
                task_name=task_name,  # ← ИСПРАВЛЕНО: было task_names
                max_sizes={"train": None, "validation": None, "test": 1},
                lr=params['lr'],
                init_prompt=params['init_prompt'],
                balance_classes=params['balance_classes'],
                prompt_length=prompt_length,
                return_results=True  # ← ДОБАВЬ этот параметр если нужно
            )
            
            # Сохраняем результаты (предполагаем что run_individual_task возвращает accuracy)
            # Если не возвращает, нужно модифицировать функцию
            my_acc = results.get('my_model_accuracy', 0)  # адаптируй под твой формат
            peft_acc = results.get('peft_model_accuracy', 0)
            
            task_results['my_model'].append(my_acc)
            task_results['peft_model'].append(peft_acc)
            
            print(f"   📈 My Model: {my_acc:.4f}, PEFT Model: {peft_acc:.4f}")
        
        all_results[task_name] = task_results
    
    # Строим графики
    plot_prompt_length_results(all_results)
    
    return all_results

def plot_prompt_length_results(all_results):
    """Строит отдельные графики для My Model и PEFT Model"""
    
    for task_name, results in all_results.items():
        lengths = results['lengths']
        my_accuracies = results['my_model']
        peft_accuracies = results['peft_model']
        
        # График 1: My Model
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(lengths, my_accuracies, 'bo-', linewidth=2, markersize=8, label='My Model')
        plt.xlabel('Prompt Length', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'My Model: Prompt Length vs Accuracy\n{task_name.upper()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Добавляем аннотации
        for i, (length, acc) in enumerate(zip(lengths, my_accuracies)):
            plt.annotate(f'{acc:.3f}', (length, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color='blue')
        
        # График 2: PEFT Model
        plt.subplot(1, 2, 2)
        plt.plot(lengths, peft_accuracies, 'ro-', linewidth=2, markersize=8, label='PEFT Model')
        plt.xlabel('Prompt Length', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'PEFT Model: Prompt Length vs Accuracy\n{task_name.upper()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Добавляем аннотации
        for i, (length, acc) in enumerate(zip(lengths, peft_accuracies)):
            plt.annotate(f'{acc:.3f}', (length, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color='red')
        
        plt.tight_layout()
        
        # Сохраняем график
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/{task_name}_prompt_length_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Graphs saved to {filename}")
        
        # Выводим сводку по лучшим длинам
        best_my_length = lengths[np.argmax(my_accuracies)]
        best_peft_length = lengths[np.argmax(peft_accuracies)]
        
        print(f"\n🏆 BEST PROMPT LENGTHS for {task_name}:")
        print(f"   My Model:  length={best_my_length}, accuracy={max(my_accuracies):.4f}")
        print(f"   PEFT Model: length={best_peft_length}, accuracy={max(peft_accuracies):.4f}")

if __name__ == "__main__":
    # Создаем папку для графиков
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Запускаем анализ
    run_prompt_length_analysis()