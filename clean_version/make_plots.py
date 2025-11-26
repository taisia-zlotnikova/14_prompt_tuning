# unified_plotter_poster.py

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib

# Настройка для качественного постера
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['figure.titlesize'] = 18

def load_all_data_files():
    """Загружает все файлы с данными"""
    data_files = [
        'init_prompt_length_data_t5-base.json',
        'init_prompt_length_data_t5-small.json', 
        'prompt_length_data_t5-base.json',
        'prompt_length_data_t5-small.json'
    ]
    
    all_data = {}
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📁 Loading: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 't5-base' in file_path:
                    model_type = 't5-base'
                else:
                    model_type = 't5-small'
                    
                if 'init_prompt' in file_path:
                    prompt_type = 'init_prompt'
                else:
                    prompt_type = 'random_prompt'
                
                for task_name, task_data in data.items():
                    key = f"{task_name}_{model_type}_{prompt_type}"
                    all_data[key] = {
                        'task_name': task_name,
                        'model_type': model_type,
                        'prompt_type': prompt_type,
                        'experiments': task_data.get('experiments', []),
                        'params': task_data.get('params', {})
                    }
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    return all_data

def group_data_by_task(all_data):
    """Группирует данные по названию задачи"""
    tasks = {}
    
    for key, data in all_data.items():
        task_name = data['task_name']
        
        if task_name not in tasks:
            tasks[task_name] = []
        
        tasks[task_name].append(data)
    
    return tasks

def plot_my_model_poster(tasks_data):
    """Строит графики для My Model с новой схемой цветов и линий"""
    
    # Новая цветовая схема: только 2 цвета + пунктир для random_prompt
    colors = {
        't5-base': '#E41A1C',    # Красный
        't5-small': '#377EB8',   # Синий
    }
    
    # Стили линий для типов промптов
    linestyles = {
        'init_prompt': '-',      # Сплошная линия
        'random_prompt': '--',   # Пунктирная линия
    }
    
    # Маркеры
    markers = {
        't5-base': 'o',      # Круг
        't5-small': 's',     # Квадрат
    }
    
    for task_name, configurations in tasks_data.items():
        print(f"\n📈 Creating My Model poster for: {task_name}")
        
        valid_configs = [config for config in configurations if config['experiments']]
        if not valid_configs:
            continue
        
        # Один график для My Model
        plt.figure(figsize=(12, 8))
        
        all_lengths = set()
        
        for config in valid_configs:
            model_type = config['model_type']
            prompt_type = config['prompt_type']
            
            color = colors[model_type]
            marker = markers[model_type]
            linestyle = linestyles[prompt_type]
            
            experiments = config['experiments']
            experiments.sort(key=lambda x: x['prompt_length'])
            
            lengths = [exp['prompt_length'] for exp in experiments]
            my_accuracies = [exp['my_accuracy'] for exp in experiments]
            
            all_lengths.update(lengths)
            
            # Создаем подпись в новом формате
            if prompt_type == 'init_prompt':
                label = f"{model_type} (init)"
            else:
                label = f"{model_type} (random)"
            
            # Рисуем линию с новыми параметрами
            plt.plot(lengths, my_accuracies, 
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=3, markersize=10, alpha=0.9, label=label)
        
        # Настройка графика
        plt.xlabel('Prompt Length', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
        plt.title(f'My Model: {task_name.upper()}', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.xticks(sorted(all_lengths), fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0, 1.0)
        
        # Улучшаем внешний вид
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Сохраняем
        os.makedirs("poster_plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poster_plots/{task_name}_my_model_poster_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ My Model poster saved: {filename}")

def plot_all_tasks_comparison(tasks_data):
    """Строит все задачи на одном большом графике с новой схемой"""
    
    # Та же цветовая схема
    colors = {
        't5-base': '#E41A1C',    # Красный
        't5-small': '#377EB8',   # Синий
    }
    
    linestyles = {
        'init_prompt': '-',      # Сплошная линия
        'random_prompt': '--',   # Пунктирная линия
    }
    
    markers = {
        't5-base': 'o',      # Круг
        't5-small': 's',     # Квадрат
    }
    
    # Создаем большой график со всеми задачами
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.flatten()
    
    tasks = list(tasks_data.keys())
    
    for idx, (ax, task_name) in enumerate(zip(axes, tasks)):
        if task_name not in tasks_data or task_name == 'boolq':
            ax.axis('off')
            continue
            
        configurations = tasks_data[task_name]
        valid_configs = [config for config in configurations if config['experiments']]
        
        if not valid_configs:
            ax.axis('off')
            continue
        
        all_lengths = set()
        
        for config in valid_configs:
            model_type = config['model_type']
            prompt_type = config['prompt_type']
            
            color = colors[model_type]
            marker = markers[model_type]
            linestyle = linestyles[prompt_type]
            
            experiments = config['experiments']
            experiments.sort(key=lambda x: x['prompt_length'])
            
            lengths = [exp['prompt_length'] for exp in experiments]
            my_accuracies = [exp['my_accuracy'] for exp in experiments]
            
            all_lengths.update(lengths)
            
            # Рисуем линию
            ax.plot(lengths, my_accuracies, color=color, marker=marker,
                   linestyle=linestyle, linewidth=2, markersize=6, alpha=0.9)
        
        # Настройка подграфика
        ax.set_xlabel('Prompt Length', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{task_name.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_xticks(sorted(all_lengths))
        ax.set_ylim(0, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Скрываем пустые subplots
    for idx in range(len(tasks), len(axes)):
        axes[idx].axis('off')
    
    # Добавляем общую легенду с новой схемой
    legend_elements = []
    
    # Добавляем элементы для всех комбинаций
    for model_type, color in colors.items():
        for prompt_type, linestyle in linestyles.items():
            label = f"{model_type} ({prompt_type})"
            marker = markers[model_type]
            
            legend_elements.append(
                plt.Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
                          linewidth=3, markersize=8, label=label)
            )
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, 0.02), fontsize=12, framealpha=0.9)
    
    plt.suptitle('My Model: Prompt Tuning Performance Across All Tasks', 
                fontsize=24, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Сохраняем
    os.makedirs("poster_plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"poster_plots/ALL_TASKS_MY_MODEL_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Comprehensive My Model comparison saved: {filename}")

def create_summary_table(tasks_data):
    """Создает красивую сводную таблицу"""
    
    print("\n" + "="*100)
    print("🏆 MY MODEL - BEST RESULTS SUMMARY")
    print("="*100)
    print(f"{'Task':<10} {'Configuration':<25} {'Best Length':<12} {'Accuracy':<10}")
    print("-"*100)
    
    for task_name, configurations in tasks_data.items():
        print(f"\n{task_name.upper():<10}")
        print("-" * 50)
        
        for config in configurations:
            model_type = config['model_type']
            prompt_type = config['prompt_type']
            experiments = config['experiments']
            
            if not experiments:
                continue
            
            # Находим лучший результат для My Model
            best_exp = max(experiments, key=lambda x: x['my_accuracy'])
            
            config_name = f"{model_type} ({prompt_type})"
            print(f"  {config_name:<25} {best_exp['prompt_length']:<12} "
                  f"{best_exp['my_accuracy']:<10.4f}")

if __name__ == "__main__":
    print("🚀 My Model Poster Analysis")
    print("="*50)
    
    # Загружаем все данные
    all_data = load_all_data_files()
    
    if not all_data:
        print("❌ No data files found or loaded")
        exit(1)
    
    print(f"\n✅ Loaded {len(all_data)} configurations")
    
    # Группируем по задачам
    tasks_data = group_data_by_task(all_data)
    
    print(f"\n📋 Tasks found: {list(tasks_data.keys())}")
    
    # Строим графики в стиле постеров
    # plot_my_model_poster(tasks_data)
    plot_all_tasks_comparison(tasks_data)
    
    # Красивая сводка
    create_summary_table(tasks_data)
    
    print(f"\n🎉 All My Model poster plots saved to 'poster_plots/' directory!")
    print("   Color Scheme:")
    print("   🔴 t5-base + init_prompt  (solid line)")
    print("   🔴 t5-base + random_prompt (dashed line)") 
    print("   🔵 t5-small + init_prompt  (solid line)")
    print("   🔵 t5-small + random_prompt (dashed line)")