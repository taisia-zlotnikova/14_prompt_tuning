# unified_plotter.py

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

def get_unified_color_scheme():
    """Единая цветовая схема для всех графиков"""
    # Красивые, контрастные цвета для постеров
    colors = {
        't5-base_init_prompt': '#E41A1C',    # Красный
        't5-base_random_prompt': '#377EB8',   # Синий
        't5-small_init_prompt': '#4DAF4A',    # Зеленый
        't5-small_random_prompt': '#984EA3',  # Фиолетовый
    }
    
    # Маркеры для лучшей различимости
    markers = {
        't5-base_init_prompt': 'o',      # Круг
        't5-base_random_prompt': 's',    # Квадрат
        't5-small_init_prompt': '^',     # Треугольник
        't5-small_random_prompt': 'D',   # Ромб
    }
    
    # Стили линий для My vs PEFT
    linestyles = {
        'my_model': '-',     # Сплошная линия
        'peft_model': '--',  # Пунктирная линия
    }
    
    return colors, markers, linestyles

def plot_task_comparison_poster(tasks_data):
    """Строит графики в стиле постеров - все на одном графике"""
    
    colors, markers, linestyles = get_unified_color_scheme()
    
    for task_name, configurations in tasks_data.items():
        print(f"\n📊 Creating poster plot for: {task_name}")
        
        valid_configs = [config for config in configurations if config['experiments']]
        if not valid_configs:
            continue
            
        # Создаем большой график для постера
        plt.figure(figsize=(16, 10))
        
        # Собираем все длины промптов для правильных подписей оси X
        all_lengths = set()
        
        for config in valid_configs:
            config_key = f"{config['model_type']}_{config['prompt_type']}"
            color = colors[config_key]
            marker = markers[config_key]
            
            experiments = config['experiments']
            experiments.sort(key=lambda x: x['prompt_length'])
            
            lengths = [exp['prompt_length'] for exp in experiments]
            my_accuracies = [exp['my_accuracy'] for exp in experiments]
            peft_accuracies = [exp['peft_accuracy'] for exp in experiments]
            
            all_lengths.update(lengths)
            
            # My Model - сплошная линия
            label_my = f"My {config['model_type']} ({config['prompt_type']})"
            plt.plot(lengths, my_accuracies, 
                    color=color, marker=marker, linestyle=linestyles['my_model'],
                    linewidth=3, markersize=10, alpha=0.9, label=label_my)
            
            # PEFT Model - пунктирная линия
            label_peft = f"PEFT {config['model_type']} ({config['prompt_type']})"
            plt.plot(lengths, peft_accuracies,
                    color=color, marker=marker, linestyle=linestyles['peft_model'], 
                    linewidth=3, markersize=10, alpha=0.9, label=label_peft)
        
        # Настройка графика
        plt.xlabel('Prompt Length', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
        plt.title(f'Task: {task_name.upper()}\nPrompt Tuning Performance Comparison', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        
        # Настройка осей
        plt.xticks(sorted(all_lengths), fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0, 1.0)
        
        # Улучшаем внешний вид
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Сохраняем в высоком качестве для постера
        os.makedirs("poster_plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poster_plots/{task_name}_poster_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Poster plot saved: {filename}")

def plot_separate_models_poster(tasks_data):
    """Строит отдельные графики для My Model и PEFT Model в стиле постеров"""
    
    colors, markers, linestyles = get_unified_color_scheme()
    
    for task_name, configurations in tasks_data.items():
        print(f"\n📈 Creating separate poster plots for: {task_name}")
        
        valid_configs = [config for config in configurations if config['experiments']]
        if not valid_configs:
            continue
        
        # Два графика рядом
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        all_lengths = set()
        
        for config in valid_configs:
            config_key = f"{config['model_type']}_{config['prompt_type']}"
            color = colors[config_key]
            marker = markers[config_key]
            
            experiments = config['experiments']
            experiments.sort(key=lambda x: x['prompt_length'])
            
            lengths = [exp['prompt_length'] for exp in experiments]
            my_accuracies = [exp['my_accuracy'] for exp in experiments]
            peft_accuracies = [exp['peft_accuracy'] for exp in experiments]
            
            all_lengths.update(lengths)
            label = f"{config['model_type']} ({config['prompt_type']})"
            
            # My Model
            ax1.plot(lengths, my_accuracies, color=color, marker=marker, 
                    linestyle=linestyles['my_model'], linewidth=3, markersize=10,
                    label=label, alpha=0.9)
        
        # Настройка My Model графика
        ax1.set_xlabel('Prompt Length', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
        ax1.set_title(f'My Model: {task_name.upper()}', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(framealpha=0.9)
        ax1.set_xticks(sorted(all_lengths))
        ax1.set_ylim(0, 1.0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Сохраняем
        os.makedirs("poster_plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poster_plots/{task_name}_separate_poster_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Separate poster plots saved: {filename}")

def plot_comprehensive_comparison(tasks_data):
    """Строит все задачи на одном большом графике для общего сравнения"""
    
    colors, markers, linestyles = get_unified_color_scheme()
    
    # Создаем большой график со всеми задачами
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))  # 2x3 сетка
    axes = axes.flatten()
    
    tasks = list(tasks_data.keys())
    
    for idx, (ax, task_name) in enumerate(zip(axes, tasks)):
        if task_name not in tasks_data:
            ax.axis('off')
            continue
            
        configurations = tasks_data[task_name]
        valid_configs = [config for config in configurations if config['experiments']]
        
        if not valid_configs:
            ax.axis('off')
            continue
        
        all_lengths = set()
        
        for config in valid_configs:
            config_key = f"{config['model_type']}_{config['prompt_type']}"
            color = colors[config_key]
            marker = markers[config_key]
            
            experiments = config['experiments']
            experiments.sort(key=lambda x: x['prompt_length'])
            
            lengths = [exp['prompt_length'] for exp in experiments]
            my_accuracies = [exp['my_accuracy'] for exp in experiments]
            peft_accuracies = [exp['peft_accuracy'] for exp in experiments]
            
            all_lengths.update(lengths)
            
            # My Model
            ax.plot(lengths, my_accuracies, color=color, marker=marker,
                   linestyle=linestyles['my_model'], linewidth=2, markersize=6,
                   alpha=0.9)
            
            # PEFT Model
            ax.plot(lengths, peft_accuracies, color=color, marker=marker,
                   linestyle=linestyles['peft_model'], linewidth=2, markersize=6,
                   alpha=0.9)
        
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
    
    # Добавляем общую легенду
    legend_elements = []
    for config_key, color in colors.items():
        legend_elements.append(plt.Line2D([0], [0], color=color, marker=markers[config_key],
                                        linestyle=linestyles['my_model'], linewidth=3,
                                        markersize=8, label=config_key.replace('_', ' ')))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               bbox_to_anchor=(0.5, 0.02), fontsize=12, framealpha=0.9)
    
    plt.suptitle('Prompt Tuning Performance Across All Tasks', 
                fontsize=24, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Сохраняем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"poster_plots/ALL_TASKS_COMPARISON_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Comprehensive comparison saved: {filename}")

def create_poster_summary(tasks_data):
    """Создает красивую сводную таблицу для постера"""
    
    print("\n" + "="*120)
    print("🏆 BEST RESULTS SUMMARY - Poster Quality")
    print("="*120)
    print(f"{'Task':<12} {'Configuration':<25} {'Best Length':<12} {'My Acc':<10} {'PEFT Acc':<10} {'Winner':<10}")
    print("-"*120)
    
    for task_name, configurations in tasks_data.items():
        print(f"\n{task_name.upper():<12}")
        print("-" * 50)
        
        for config in configurations:
            model_type = config['model_type']
            prompt_type = config['prompt_type']
            experiments = config['experiments']
            
            if not experiments:
                continue
            
            best_my = max(experiments, key=lambda x: x['my_accuracy'])
            best_peft = max(experiments, key=lambda x: x['peft_accuracy'])
            
            diff = best_my['my_accuracy'] - best_peft['peft_accuracy']
            
            if diff > 0.01:
                winner = "My Model"
            elif diff < -0.01:
                winner = "PEFT"
            else:
                winner = "Tie"
            
            config_name = f"{model_type} ({prompt_type})"
            print(f"  {config_name:<25} {best_my['prompt_length']:<12} "
                  f"{best_my['my_accuracy']:<10.4f} {best_peft['peft_accuracy']:<10.4f} {winner:<10}")

if __name__ == "__main__":
    print("🚀 Unified Data Analysis - Poster Quality")
    print("="*60)
    
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
    # plot_task_comparison_poster(tasks_data)
    plot_separate_models_poster(tasks_data)
    # plot_comprehensive_comparison(tasks_data)
    
    # Красивая сводка
    create_poster_summary(tasks_data)
    
    print(f"\n🎉 All poster-quality plots saved to 'poster_plots/' directory!")
    print("   Colors are consistent across all plots:")
    print("   🔴 t5-base + init_prompt")
    print("   🔵 t5-base + random_prompt") 
    print("   🟢 t5-small + init_prompt")
    print("   🟣 t5-small + random_prompt")
    print("   My Model: Solid lines, PEFT Model: Dashed lines")