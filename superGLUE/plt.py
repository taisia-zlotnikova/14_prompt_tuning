import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


class SuperGLUEComparison:
    """Класс для визуализации сравнения подходов на SuperGLUE"""
    
    def __init__(self):
        """Инициализация данных из статьи"""
        # Модели
        self.models = ['T5-Small', 'T5-Base']
        self.model_params = [60.5e6, 223e6]  # Параметры моделей
        
        # === ACCURACY (%) на SuperGLUE ===
        self.fine_tuning_acc = [78, 82]
        self.prompt_tuning_acc = [66, 72]
        self.prompt_design_acc = [63, 63]
        
        # === F1-SCORE на SuperGLUE ===
        self.fine_tuning_f1 = [0.76, 0.80]
        self.prompt_tuning_f1 = [0.64, 0.70]
        self.prompt_design_f1 = [0.61, 0.61]
        
        # Цвета для графиков
        self.colors = {
            'fine_tuning': '#2E86AB',    # Синий
            'prompt_tuning': '#A23B72',  # Пурпурный
            'prompt_design': '#F18F01'   # Оранжевый
        }
    
    def plot_comparison(self, save_path: str = 'superglue_comparison_t5.png') -> None:
        """
        Построить сравнительные графики Accuracy и F1-Score
        
        Args:
            save_path: путь для сохранения графика
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # === ГРАФИК 1: ACCURACY ===
        ax1.plot(self.models, self.fine_tuning_acc, 'o-', linewidth=2.5, markersize=8,
                 label='Fine-tuning', color=self.colors['fine_tuning'])
        ax1.plot(self.models, self.prompt_tuning_acc, 's-', linewidth=2.5, markersize=8,
                 label='Prompt Tuning', color=self.colors['prompt_tuning'])
        ax1.plot(self.models, self.prompt_design_acc, '^-', linewidth=2.5, markersize=8,
                 label='Prompt Design', color=self.colors['prompt_design'])
        
        # Оформление
        ax1.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('SuperGLUE Accuracy Comparison', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.set_ylim([60, 85])
        
        # Добавляем значения на графике
        for i, model in enumerate(self.models):
            ax1.text(i, self.fine_tuning_acc[i] + 0.8, f'{self.fine_tuning_acc[i]}%',
                     ha='center', fontsize=10, fontweight='bold')
            ax1.text(i, self.prompt_tuning_acc[i] - 1.5, f'{self.prompt_tuning_acc[i]}%',
                     ha='center', fontsize=10, fontweight='bold')
            ax1.text(i, self.prompt_design_acc[i] - 1.5, f'{self.prompt_design_acc[i]}%',
                     ha='center', fontsize=10, fontweight='bold')
        
        # === ГРАФИК 2: F1-SCORE ===
        ax2.plot(self.models, self.fine_tuning_f1, 'o-', linewidth=2.5, markersize=8,
                 label='Fine-tuning', color=self.colors['fine_tuning'])
        ax2.plot(self.models, self.prompt_tuning_f1, 's-', linewidth=2.5, markersize=8,
                 label='Prompt Tuning', color=self.colors['prompt_tuning'])
        ax2.plot(self.models, self.prompt_design_f1, '^-', linewidth=2.5, markersize=8,
                 label='Prompt Design', color=self.colors['prompt_design'])
        
        # Оформление
        ax2.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax2.set_title('SuperGLUE F1-Score Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11, loc='lower right')
        ax2.set_ylim([0.60, 0.85])
        
        # Добавляем значения на графике
        for i, model in enumerate(self.models):
            ax2.text(i, self.fine_tuning_f1[i] + 0.01, f'{self.fine_tuning_f1[i]:.2f}',
                     ha='center', fontsize=10, fontweight='bold')
            ax2.text(i, self.prompt_tuning_f1[i] - 0.02, f'{self.prompt_tuning_f1[i]:.2f}',
                     ha='center', fontsize=10, fontweight='bold')
            ax2.text(i, self.prompt_design_f1[i] - 0.02, f'{self.prompt_design_f1[i]:.2f}',
                     ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ График сохранён как '{save_path}'")
        plt.show()
    
    def print_analysis(self) -> None:
        """Вывести подробный анализ результатов"""
        print("\n" + "=" * 80)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ: T5-Small vs T5-Base на SuperGLUE CB")
        print("=" * 80)
        
        # === ТАБЛИЦА ACCURACY ===
        print("\n📊 ACCURACY (%):")
        print("-" * 80)
        print(f"{'Модель':<15} {'Fine-tuning':<15} {'Prompt Tuning':<15} {'Prompt Design':<15} {'Разрыв (PT-FT)':<15}")
        print("-" * 80)
        
        for i, model in enumerate(self.models):
            gap = self.prompt_tuning_acc[i] - self.fine_tuning_acc[i]
            print(f"{model:<15} {self.fine_tuning_acc[i]:<15} {self.prompt_tuning_acc[i]:<15} {self.prompt_design_acc[i]:<15} {gap:<15.1f}")
        
        # === ТАБЛИЦА F1 ===
        print("\n📈 F1-SCORE:")
        print("-" * 80)
        print(f"{'Модель':<15} {'Fine-tuning':<15} {'Prompt Tuning':<15} {'Prompt Design':<15} {'Разрыв (PT-FT)':<15}")
        print("-" * 80)
        
        for i, model in enumerate(self.models):
            gap = self.prompt_tuning_f1[i] - self.fine_tuning_f1[i]
            print(f"{model:<15} {self.fine_tuning_f1[i]:<15.3f} {self.prompt_tuning_f1[i]:<15.3f} {self.prompt_design_f1[i]:<15.3f} {gap:<15.3f}")
        
        # === ВЫВОДЫ ===
        print("\n" + "=" * 80)
        print("КЛЮЧЕВЫЕ ВЫВОДЫ")
        print("=" * 80)
        
        print(f"""
1. 🎯 ТРЕНД ACCURACY:
   • T5-Small: Fine-tuning (78%) >> Prompt Tuning (66%) >> Prompt Design (63%)
   • T5-Base:  Fine-tuning (82%) > Prompt Tuning (72%) >> Prompt Design (63%)
   
   ⚠️ Разрыв между Fine-tuning и Prompt Tuning:
   • T5-Small: -12% (критичный)
   • T5-Base:  -10% (всё ещё значительный)
   
   💡 Вывод: На маленьких моделях Prompt Tuning всё ещё заметно отстаёт

2. 📈 ТРЕНД F1-SCORE:
   • T5-Small: Fine-tuning (0.76) >> Prompt Tuning (0.64) >> Prompt Design (0.61)
   • T5-Base:  Fine-tuning (0.80) > Prompt Tuning (0.70) >> Prompt Design (0.61)
   
   ⚠️ Разрыв F1:
   • T5-Small: -0.12 (12%)
   • T5-Base:  -0.10 (10%)

3. 🚀 УЛУЧШЕНИЕ PROMPT TUNING:
   • От T5-Small к T5-Base: +6% accuracy
   • От T5-Small к T5-Base: +0.06 F1
   
   💡 Prompt Tuning улучшается БЫСТРЕЕ, чем Fine-tuning при увеличении модели!

4. 🔧 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:

   ✅ ИСПОЛЬЗУЙТЕ Fine-tuning, если:
      • Нужна максимальная точность (78-82%)
      • Есть достаточно вычислительных ресурсов
      • Нужно только для одной задачи
   
   ✅ ИСПОЛЬЗУЙТЕ Prompt Tuning, если:
      • Нужен баланс качество/скорость (66-72%)
      • Нужно обслуживать много задач одной моделью
      • Ограничены память и время обучения
      • Планируете масштабировать на T5-Large/XL
   
   ❌ ИЗБЕГАЙТЕ Prompt Design, если:
      • Нужна хорошая точность (только 63%)
      • Лучше использовать Prompt Tuning вместо manual prompting

5. 📊 ИНТЕРПОЛЯЦИЯ НА T5-XXL (11B параметров):
   На основе тренда из статьи:
   • Fine-tuning: ~89%
   • Prompt Tuning: ~89% (почти совпадает!)
   • Prompt Design: ~71% (даже с GPT-3 175B)
   
   💡 Вывод: На больших моделях разрыв исчезает!

6. 📚 ССЫЛКА НА СТАТЬЮ:
   Lester, B., Al-Rfou, R., & Constant, N. (2021)
   "The Power of Scale for Parameter-Efficient Prompt Tuning"
   arXiv:2104.08691
""")
        
        print("=" * 80)
    
    def get_metrics_table(self) -> Dict[str, Dict[str, float]]:
        """Получить метрики в виде словаря"""
        metrics = {}
        
        for i, model in enumerate(self.models):
            metrics[model] = {
                'fine_tuning_acc': self.fine_tuning_acc[i],
                'prompt_tuning_acc': self.prompt_tuning_acc[i],
                'prompt_design_acc': self.prompt_design_acc[i],
                'fine_tuning_f1': self.fine_tuning_f1[i],
                'prompt_tuning_f1': self.prompt_tuning_f1[i],
                'prompt_design_f1': self.prompt_design_f1[i],
                'acc_gap': self.prompt_tuning_acc[i] - self.fine_tuning_acc[i],
                'f1_gap': self.prompt_tuning_f1[i] - self.fine_tuning_f1[i],
            }
        
        return metrics


# ============================================================================
# ИСПОЛЬЗОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    # Создаём объект сравнения
    comparison = SuperGLUEComparison()
    
    # Строим графики
    print("🚀 Построение графиков...")
    comparison.plot_comparison(save_path='superglue_comparison_t5.png')
    
    # Выводим анализ
    comparison.print_analysis()
    
    # Получаем метрики в виде словаря
    metrics = comparison.get_metrics_table()
    print("\n📋 Метрики в виде словаря:")
    for model, data in metrics.items():
        print(f"\n{model}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
