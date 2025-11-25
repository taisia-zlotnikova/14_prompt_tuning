# plot_generator.py

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

name_model = os.getenv('NAME_MODEL', 't5-small')
data_file_name = f"prompt_length_data_{name_model}.json"
plots_dir = f'plots_{name_model}'

def load_data():
    """Загружает данные из файла"""
    data_file = data_file_name
    
    if not os.path.exists(data_file):
        print("❌ No data file found. Run data_collector.py first.")
        return None
    
    with open(data_file, 'r') as f:
        return json.load(f)

def plot_combined_results(all_data):
    """Строит графики с обеими моделями на одном графике для каждой задачи"""
    
    for task_name, task_data in all_data.items():
        experiments = task_data['experiments']
        
        if not experiments:
            print(f"⚠️  No data for {task_name}")
            continue
        
        # Сортируем эксперименты по длине промпта
        experiments.sort(key=lambda x: x['prompt_length'])
        
        lengths = [exp['prompt_length'] for exp in experiments]
        my_accuracies = [exp['my_accuracy'] for exp in experiments]
        peft_accuracies = [exp['peft_accuracy'] for exp in experiments]
        
        # Создаем график
        plt.figure(figsize=(10, 6))
        
        # My Model - синяя линия
        plt.plot(lengths, my_accuracies, 'bo-', linewidth=2, markersize=8, 
                label='My Model', alpha=0.8)
        
        # PEFT Model - красная линия  
        plt.plot(lengths, peft_accuracies, 'ro-', linewidth=2, markersize=8,
                label='PEFT Model', alpha=0.8)
        
        plt.xlabel('Prompt Length', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Prompt Length vs Accuracy: {task_name.upper()}\n'
                 f"LR={task_data['params']['lr']}, Prompt='{task_data['params']['init_prompt'][:20]}...'", 
                 fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Добавляем аннотации для обеих моделей
        for i, (length, my_acc, peft_acc) in enumerate(zip(lengths, my_accuracies, peft_accuracies)):
            # Аннотации для My Model (сверху)
            plt.annotate(f'My: {my_acc:.3f}', (length, my_acc), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=8, color='blue', alpha=0.8)
            
            # Аннотации для PEFT Model (снизу)
            plt.annotate(f'PEFT: {peft_acc:.3f}', (length, peft_acc),
                        textcoords="offset points", xytext=(0,-15),
                        ha='center', fontsize=8, color='red', alpha=0.8)
        
        # Настройка внешнего вида
        plt.xticks(lengths)
        plt.ylim(bottom=0)
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{plots_dir}", exist_ok=True)
        filename = f"{plots_dir}/{task_name}_combined_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Graph saved: {filename}")
        
        # Выводим статистику
        best_my_idx = np.argmax(my_accuracies)
        best_peft_idx = np.argmax(peft_accuracies)
        
        print(f"🏆 {task_name.upper()} - Best lengths:")
        print(f"   My Model:  length={lengths[best_my_idx]}, accuracy={my_accuracies[best_my_idx]:.4f}")
        print(f"   PEFT Model: length={lengths[best_peft_idx]}, accuracy={peft_accuracies[best_peft_idx]:.4f}")

def plot_comparison_across_tasks(all_data):
    """Строит график сравнения лучших accuracy по всем задачам"""
    
    tasks = []
    best_my_accuracies = []
    best_peft_accuracies = []
    best_my_lengths = []
    best_peft_lengths = []
    
    for task_name, task_data in all_data.items():
        if not task_data['experiments']:
            continue
            
        experiments = task_data['experiments']
        my_accuracies = [exp['my_accuracy'] for exp in experiments]
        peft_accuracies = [exp['peft_accuracy'] for exp in experiments]
        lengths = [exp['prompt_length'] for exp in experiments]
        
        best_my_idx = np.argmax(my_accuracies)
        best_peft_idx = np.argmax(peft_accuracies)
        
        tasks.append(task_name)
        best_my_accuracies.append(my_accuracies[best_my_idx])
        best_peft_accuracies.append(peft_accuracies[best_peft_idx])
        best_my_lengths.append(lengths[best_my_idx])
        best_peft_lengths.append(lengths[best_peft_idx])
    
    if not tasks:
        print("❌ No data to plot")
        return
    
    # Столбчатая диаграмма сравнения
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, best_my_accuracies, width, label='My Model', alpha=0.8, color='blue')
    bars2 = plt.bar(x + width/2, best_peft_accuracies, width, label='PEFT Model', alpha=0.8, color='red')
    
    plt.xlabel('Tasks', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.title('Best Accuracy Comparison Across Tasks', fontsize=14)
    plt.xticks(x, [task.upper() for task in tasks])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения и длины промптов на столбцы
    for i, (my_acc, peft_acc, my_len, peft_len) in enumerate(zip(
        best_my_accuracies, best_peft_accuracies, best_my_lengths, best_peft_lengths)):
        
        plt.text(i - width/2, my_acc + 0.01, f'{my_acc:.3f}\n(len={my_len})', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, peft_acc + 0.01, f'{peft_acc:.3f}\n(len={peft_len})', 
                ha='center', va='bottom', fontsize=8)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plots_dir}/task_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison graph saved: {filename}")

def plot_optimal_lengths_heatmap(all_data):
    """Строит heatmap оптимальных длин промптов"""
    
    tasks = list(all_data.keys())
    lengths = sorted(set(exp['prompt_length'] for task_data in all_data.values() 
                        for exp in task_data['experiments']))
    
    if not lengths:
        print("❌ No length data available")
        return
    
    # Создаем матрицы для heatmap
    my_accuracy_matrix = np.zeros((len(tasks), len(lengths)))
    peft_accuracy_matrix = np.zeros((len(tasks), len(lengths)))
    
    for i, task_name in enumerate(tasks):
        task_data = all_data[task_name]
        for exp in task_data['experiments']:
            j = lengths.index(exp['prompt_length'])
            my_accuracy_matrix[i, j] = exp['my_accuracy']
            peft_accuracy_matrix[i, j] = exp['peft_accuracy']
    
    # Heatmap для My Model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = ax1.imshow(my_accuracy_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_title('My Model Accuracy Heatmap')
    ax1.set_xlabel('Prompt Length')
    ax1.set_ylabel('Task')
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels(lengths)
    ax1.set_yticks(range(len(tasks)))
    ax1.set_yticklabels([task.upper() for task in tasks])
    plt.colorbar(im1, ax=ax1)
    
    # Heatmap для PEFT Model
    im2 = ax2.imshow(peft_accuracy_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_title('PEFT Model Accuracy Heatmap')
    ax2.set_xlabel('Prompt Length')
    ax2.set_ylabel('Task')
    ax2.set_xticks(range(len(lengths)))
    ax2.set_xticklabels(lengths)
    ax2.set_yticks(range(len(tasks)))
    ax2.set_yticklabels([task.upper() for task in tasks])
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plots_dir}/heatmap_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Heatmap saved: {filename}")

if __name__ == "__main__":
    # Загружаем данные
    data = load_data()
    
    if data:
        # Строим все графики
        plot_combined_results(data)
        plot_comparison_across_tasks(data)
        plot_optimal_lengths_heatmap(data)
        
        print("\n✅ All plots generated!")