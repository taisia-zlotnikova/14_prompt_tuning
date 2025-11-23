"""
Анализ результатов: Overfitting Analysis для Model Tuning vs Prompt Tuning
Вычисление метрик переобучения и distribution shift
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


class OverfittingAnalyzer:
    """Анализ переобучения и distribution shift"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results = {}
        
    def load_results(self, experiment_name: str):
        """Загружает результаты эксперимента"""
        results_file = self.results_dir / experiment_name / "results.json"
        with open(results_file) as f:
            self.results[experiment_name] = json.load(f)
    
    def compute_train_test_gap(self, 
                               experiment_name: str,
                               train_dataset: str,
                               eval_datasets: List[str]) -> Dict[str, float]:
        """
        Вычисляет gap между train и eval метриками
        High gap = высокое переобучение
        """
        metrics = self.results[experiment_name]["metrics"]
        gaps = {}
        
        for eval_ds in eval_datasets:
            for metric in metrics.get(eval_ds, {}):
                # Логируем: берем first eval value как proxy для "train" метрики
                # В реальности нужно отслеживать train метрики отдельно
                train_val = metrics.get(train_dataset, {}).get(metric, [0])
                eval_val = metrics.get(eval_ds, {}).get(metric, [0])
                
                if isinstance(train_val, list):
                    train_val = np.mean(train_val) if train_val else 0
                if isinstance(eval_val, list):
                    eval_val = np.mean(eval_val) if eval_val else 0
                
                gap = abs(train_val - eval_val)
                gaps[f"{eval_ds}_{metric}_gap"] = gap
        
        return gaps
    
    def compute_distribution_shift_impact(self,
                                         experiment_name: str,
                                         in_dist_dataset: str,
                                         ood_datasets: List[str]) -> Dict[str, float]:
        """
        Вычисляет падение производительности при distribution shift
        - In-distribution: датасет, на котором обучали
        - Out-of-distribution: датасеты с другим распределением
        
        Metric: (in_dist_perf - ood_perf) / in_dist_perf * 100 (%)
        """
        metrics = self.results[experiment_name]["metrics"]
        shifts = {}
        
        for metric_name in metrics.get(in_dist_dataset, {}):
            in_dist_perf = np.mean(metrics[in_dist_dataset][metric_name])
            
            for ood_ds in ood_datasets:
                if ood_ds in metrics and metric_name in metrics[ood_ds]:
                    ood_perf = np.mean(metrics[ood_ds][metric_name])
                    
                    # Нормализованное падение (в процентах)
                    if in_dist_perf > 0:
                        drop_pct = (in_dist_perf - ood_perf) / in_dist_perf * 100
                    else:
                        drop_pct = 0
                    
                    shifts[f"{ood_ds}_{metric_name}_drop_pct"] = drop_pct
        
        return shifts
    
    def compute_spurious_correlation_proxy(self,
                                          experiment_name: str) -> Dict[str, float]:
        """
        Прокси для детектирования ложных корреляций:
        Если модель сильно переобучилась, то она найдет spurious correlations
        в тренировочном датасете, которые не работают на других датасетах.
        
        Используем статистику: large gap between different test sets on same model
        = признак того, что модель выучила dataset-specific features
        """
        metrics = self.results[experiment_name]["metrics"]
        all_scores = {}
        
        # Собираем все eval датасеты
        eval_datasets = list(metrics.keys())
        
        for metric_name in ["exact_match", "f1", "accuracy"]:
            metric_scores = []
            
            for ds in eval_datasets:
                if metric_name in metrics.get(ds, {}):
                    score = np.mean(metrics[ds][metric_name])
                    metric_scores.append(score)
            
            if metric_scores:
                # Коэффициент вариации: высокий CV = разные датасеты сильно отличаются
                cv = np.std(metric_scores) / (np.mean(metric_scores) + 1e-8)
                all_scores[f"{metric_name}_cv"] = cv
        
        return all_scores
    
    def compare_tuning_approaches(self,
                                 model_tuning_exp: str,
                                 prompt_tuning_exp: str) -> pd.DataFrame:
        """
        Сравнивает model tuning vs prompt tuning по метрикам переобучения
        """
        self.load_results(model_tuning_exp)
        self.load_results(prompt_tuning_exp)
        
        # Загружаем конфиги для получения dataset info
        model_config = self.results[model_tuning_exp]["config"]
        
        train_ds = model_config["train_dataset"]
        eval_dss = model_config["eval_datasets"]
        
        # Вычисляем метрики для обоих подходов
        mt_gaps = self.compute_train_test_gap(model_tuning_exp, train_ds, eval_dss)
        pt_gaps = self.compute_train_test_gap(prompt_tuning_exp, train_ds, eval_dss)
        
        mt_shift = self.compute_distribution_shift_impact(
            model_tuning_exp, train_ds, eval_dss
        )
        pt_shift = self.compute_distribution_shift_impact(
            prompt_tuning_exp, train_ds, eval_dss
        )
        
        mt_spurious = self.compute_spurious_correlation_proxy(model_tuning_exp)
        pt_spurious = self.compute_spurious_correlation_proxy(prompt_tuning_exp)
        
        # Собираем в DataFrame
        comparison_data = {
            "Model Tuning - Gap": mt_gaps,
            "Prompt Tuning - Gap": pt_gaps,
            "Model Tuning - Shift": mt_shift,
            "Prompt Tuning - Shift": pt_shift,
            "Model Tuning - Spurious": mt_spurious,
            "Prompt Tuning - Spurious": pt_spurious,
        }
        
        df = pd.DataFrame.from_dict(comparison_data, orient="columns")
        return df
    
    def visualize_distribution_shift(self,
                                    experiment_name: str,
                                    output_file: str = None):
        """
        Визуализирует падение производительности при distribution shift
        """
        self.load_results(experiment_name)
        
        metrics = self.results[experiment_name]["metrics"]
        config = self.results[experiment_name]["config"]
        
        train_ds = config["train_dataset"]
        eval_dss = config["eval_datasets"]
        
        # Собираем данные для графика
        plot_data = {}
        
        for metric_name in ["exact_match", "f1", "accuracy"]:
            scores = []
            labels = [train_ds]
            
            # In-distribution
            in_dist_score = np.mean(metrics.get(train_ds, {}).get(metric_name, [0]))
            scores.append(in_dist_score)
            
            # Out-of-distribution
            for eval_ds in eval_dss:
                ood_score = np.mean(metrics.get(eval_ds, {}).get(metric_name, [0]))
                scores.append(ood_score)
                labels.append(eval_ds)
            
            plot_data[metric_name] = (labels, scores)
        
        # Рисуем
        fig, axes = plt.subplots(1, len(plot_data), figsize=(15, 5))
        
        if len(plot_data) == 1:
            axes = [axes]
        
        for ax, (metric_name, (labels, scores)) in zip(axes, plot_data.items()):
            colors = ["green"] + ["red"] * (len(labels) - 1)
            bars = ax.bar(range(len(labels)), scores, color=colors, alpha=0.7)
            
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} - Distribution Shift")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylim([0, 1])
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150)
            print(f"Saved to {output_file}")
        else:
            plt.show()
    
    def generate_report(self,
                       model_tuning_exp: str,
                       prompt_tuning_exp: str,
                       output_file: str = "overfitting_report.txt"):
        """Генерирует текстовый отчет"""
        
        comparison_df = self.compare_tuning_approaches(
            model_tuning_exp, prompt_tuning_exp
        )
        
        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("OVERFITTING ANALYSIS REPORT\n")
            f.write("Model Tuning vs Prompt Tuning (Section 5)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("HYPOTHESIS:\n")
            f.write("- Model Tuning: Все параметры обновляются -> высокий риск spurious correlations\n")
            f.write("- Prompt Tuning: Frozen LM + learnable prompts -> лучшая обобщаемость\n\n")
            
            f.write("RESULTS:\n")
            f.write(str(comparison_df) + "\n\n")
            
            # Интерпретация
            f.write("INTERPRETATION:\n")
            
            # Train-Test Gap
            f.write("\n1. Train-Test Gap Analysis:\n")
            mt_avg_gap = comparison_df["Model Tuning - Gap"].mean()
            pt_avg_gap = comparison_df["Prompt Tuning - Gap"].mean()
            f.write(f"   Model Tuning avg gap: {mt_avg_gap:.4f}\n")
            f.write(f"   Prompt Tuning avg gap: {pt_avg_gap:.4f}\n")
            if pt_avg_gap < mt_avg_gap:
                f.write(f"   ✓ Prompt Tuning has LOWER gap (BETTER generalization)\n")
            else:
                f.write(f"   ✗ Model Tuning has LOWER gap\n")
            
            # Distribution Shift
            f.write("\n2. Distribution Shift Impact:\n")
            mt_avg_shift = comparison_df["Model Tuning - Shift"].mean()
            pt_avg_shift = comparison_df["Prompt Tuning - Shift"].mean()
            f.write(f"   Model Tuning avg drop: {mt_avg_shift:.2f}%\n")
            f.write(f"   Prompt Tuning avg drop: {pt_avg_shift:.2f}%\n")
            if pt_avg_shift < mt_avg_shift:
                f.write(f"   ✓ Prompt Tuning is MORE ROBUST to distribution shift\n")
            else:
                f.write(f"   ✗ Model Tuning is more robust\n")
            
            # Spurious Correlations
            f.write("\n3. Spurious Correlations (via CV metric):\n")
            mt_spurious = comparison_df["Model Tuning - Spurious"].mean()
            pt_spurious = comparison_df["Prompt Tuning - Spurious"].mean()
            f.write(f"   Model Tuning avg CV: {mt_spurious:.4f}\n")
            f.write(f"   Prompt Tuning avg CV: {pt_spurious:.4f}\n")
            if mt_spurious > pt_spurious:
                f.write(f"   ✓ Prompt Tuning has LOWER variance across datasets\n")
                f.write(f"     -> less dataset-specific overfitting\n")
            else:
                f.write(f"   ✗ Model Tuning has lower variance\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("CONCLUSION:\n")
            if pt_avg_gap < mt_avg_gap and mt_spurious > pt_spurious:
                f.write("✓ HYPOTHESIS CONFIRMED: Prompt Tuning generalizes better\n")
                f.write("  and is more resistant to spurious correlations.\n")
            else:
                f.write("✗ HYPOTHESIS NOT FULLY SUPPORTED\n")
                f.write("  May require further investigation or parameter tuning.\n")
        
        print(f"Report saved to {output_file}")


# Пример использования
if __name__ == "__main__":
    analyzer = OverfittingAnalyzer(results_dir="./results")
    
    # Сравнение двух подходов на Эксперименте 1 (QA)
    analyzer.compare_tuning_approaches(
        model_tuning_exp="qa_model_tuning_vs_prompt_tuning_mt",
        prompt_tuning_exp="qa_model_tuning_vs_prompt_tuning_pt"
    )
    
    # Визуализация distribution shift
    analyzer.visualize_distribution_shift(
        experiment_name="qa_model_tuning_vs_prompt_tuning_mt",
        output_file="mt_distribution_shift.png"
    )
    analyzer.visualize_distribution_shift(
        experiment_name="qa_model_tuning_vs_prompt_tuning_pt",
        output_file="pt_distribution_shift.png"
    )
    
    # Генерирование отчета
    analyzer.generate_report(
        model_tuning_exp="qa_model_tuning_vs_prompt_tuning_mt",
        prompt_tuning_exp="qa_model_tuning_vs_prompt_tuning_pt",
        output_file="overfitting_analysis_report.txt"
    )
