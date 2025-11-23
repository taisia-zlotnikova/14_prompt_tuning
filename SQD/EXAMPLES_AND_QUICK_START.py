"""
Примеры использования и quick-start guide для Section 5 experiments
"""

# ============================================================================
# QUICK START: Минимальный пример для запуска
# ============================================================================

"""
# 1. Установка зависимостей
pip install transformers datasets evaluate torch matplotlib seaborn pandas

# 2. Быстрый запуск (5-10 минут на CPU, ~30-40 минут на GPU)
python run_section5.py --qa-only

# 3. Просмотр результатов
cat results/qa_overfitting_report.txt
"""


# ============================================================================
# ПРИМЕР 1: Запуск только Model Tuning
# ============================================================================

from experiments_section5 import QAExperiment, ExperimentConfig

# Конфигурация для QA эксперимента с model tuning
config = ExperimentConfig(
    experiment_name="custom_qa_model_tuning",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="model_tuning",  # ← ВСЕ параметры обновляются
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"],  # OOD datasets
    max_source_length=512,
    max_target_length=50,
    batch_size=32,
    num_epochs=3,
    learning_rate=1e-3,
    warmup_steps=500,
    seed=42,
)

# Запуск эксперимента
experiment = QAExperiment(config)
results = experiment.run()

# Вывод:
# - Логируется количество trainable параметров (~220M для t5-base)
# - Сохраняются results в ./results/custom_qa_model_tuning/results.json
# - Отчет содержит: metrics, overfitting_analysis, trainable_params


# ============================================================================
# ПРИМЕР 2: Запуск только Prompt Tuning
# ============================================================================

from experiments_section5 import PromptTuningWrapper

# Конфигурация для QA эксперимента с prompt tuning
config_pt = ExperimentConfig(
    experiment_name="custom_qa_prompt_tuning",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="prompt_tuning",  # ← Только prompts обновляются
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"],
    max_source_length=512,
    max_target_length=50,
    batch_size=32,
    num_epochs=3,
    learning_rate=1e-3,
    warmup_steps=500,
    
    # Параметры prompt tuning
    prompt_length=20,  # Длина soft prompt в токенах
    init_text="question answering: ",  # Инициализация из vocabulary
)

experiment_pt = QAExperiment(config_pt)
results_pt = experiment_pt.run()

# Вывод:
# - Логируется количество trainable параметров (~50K для t5-base с prompt_length=20)
# - Замораживаются все параметры LM, обновляются только soft prompts
# - Более быстрое обучение и лучшая обобщаемость


# ============================================================================
# ПРИМЕР 3: Сравнение метрик переобучения
# ============================================================================

from analysis_overfitting import OverfittingAnalyzer

analyzer = OverfittingAnalyzer(results_dir="./results")

# Загружаем оба эксперимента
analyzer.load_results("custom_qa_model_tuning")
analyzer.load_results("custom_qa_prompt_tuning")

# 1. Вычисляем train-test gap (proxy для переобучения)
gaps_mt = analyzer.compute_train_test_gap(
    "custom_qa_model_tuning",
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"]
)
# Результат:
# {'squad_dev_exact_match_gap': 0.05, 'mrqa_newsqa_exact_match_gap': 0.15, ...}
# Интерпретация: gap=0.15 означает, что модель забыла 15% F1 на OOD данных

gaps_pt = analyzer.compute_train_test_gap(
    "custom_qa_prompt_tuning",
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"]
)
# Ожидаемо: gaps_pt < gaps_mt (prompt tuning обобщается лучше)

print(f"Model Tuning avg gap: {np.mean(list(gaps_mt.values())):.4f}")
print(f"Prompt Tuning avg gap: {np.mean(list(gaps_pt.values())):.4f}")
# ✓ Если PT gap < MT gap -> гипотеза подтверждена!


# 2. Вычисляем impact distribution shift
shift_mt = analyzer.compute_distribution_shift_impact(
    "custom_qa_model_tuning",
    in_dist_dataset="squad",
    ood_datasets=["mrqa_newsqa", "mrqa_triviaqa"]
)
# Результат:
# {'mrqa_newsqa_exact_match_drop_pct': 25.5, 'mrqa_triviaqa_exact_match_drop_pct': 32.1, ...}
# Интерпретация: 25.5% падение F1 на OOD данных

shift_pt = analyzer.compute_distribution_shift_impact(
    "custom_qa_prompt_tuning",
    in_dist_dataset="squad",
    ood_datasets=["mrqa_newsqa", "mrqa_triviaqa"]
)
# Ожидаемо: shift_pt < shift_mt (более устойчив к distribution shift)

print(f"Model Tuning avg drop: {np.mean(list(shift_mt.values())):.2f}%")
print(f"Prompt Tuning avg drop: {np.mean(list(shift_pt.values())):.2f}%")
# ✓ Если PT drop < MT drop -> модель выучила general features!


# 3. Анализ spurious correlations (через CV)
spurious_mt = analyzer.compute_spurious_correlation_proxy("custom_qa_model_tuning")
spurious_pt = analyzer.compute_spurious_correlation_proxy("custom_qa_prompt_tuning")

print(f"Model Tuning CV: {spurious_mt['exact_match_cv']:.4f}")
print(f"Prompt Tuning CV: {spurious_pt['exact_match_cv']:.4f}")
# ✓ Если PT CV < MT CV -> меньше dataset-specific overfitting!


# ============================================================================
# ПРИМЕР 4: Паraphrase Detection с cross-dataset evaluation
# ============================================================================

from experiments_section5 import ParaphraseExperiment

config_para = ExperimentConfig(
    experiment_name="custom_paraphrase_mt",
    task_type="paraphrase",
    model_name="t5-base",
    tuning_approach="model_tuning",
    train_dataset="qqp",  # Обучаемся на QQP
    eval_datasets=["qqp", "mrpc"],  # Тестируем на обоих
    batch_size=32,
    num_epochs=10,  # Больше эпох для паrafphrazing
    learning_rate=1e-3,
)

para_exp = ParaphraseExperiment(config_para)
results = para_exp.run_cross_dataset_eval()

# Вывод:
# - Модель обучается на QQP
# - Оценивается на QQP (in-distribution) и MRPC (out-of-distribution)
# - Высокая разница = переобучение; низкая = хорошо обобщается


# ============================================================================
# ПРИМЕР 5: Визуализация distribution shift
# ============================================================================

import matplotlib.pyplot as plt

analyzer = OverfittingAnalyzer(results_dir="./results")

# Визуализирует падение точности на OOD датасетах
analyzer.visualize_distribution_shift(
    experiment_name="custom_qa_model_tuning",
    output_file="mt_distribution_shift.png"
)
# График показывает: SQuAD (in-dist) vs MRQA датасеты (OOD)
# Если столбцы сильно падают -> модель переобучилась

analyzer.visualize_distribution_shift(
    experiment_name="custom_qa_prompt_tuning",
    output_file="pt_distribution_shift.png"
)
# Ожидаемо: столбцы менее «горбатые» (более плоские падения)


# ============================================================================
# ПРИМЕР 6: Генерирование полного отчета
# ============================================================================

analyzer.generate_report(
    model_tuning_exp="custom_qa_model_tuning",
    prompt_tuning_exp="custom_qa_prompt_tuning",
    output_file="final_report.txt"
)

# Отчет содержит:
# 1. Гипотезу
# 2. Сравнительные метрики
# 3. Интерпретацию результатов
# 4. Вывод: подтверждена ли гипотеза


# ============================================================================
# ПРИМЕР 7: Анализ trainable параметров
# ============================================================================

from experiments_section5 import ModelTuningWrapper, PromptTuningWrapper
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Model Tuning: все параметры обновляются
mt_wrapper = ModelTuningWrapper(model)
mt_params = mt_wrapper.get_trainable_params()
print(f"Model Tuning trainable params: {mt_params:,}")  # ~222M

# Prompt Tuning: только prompts обновляются
pt_wrapper = PromptTuningWrapper(model, prompt_length=20)
pt_params = pt_wrapper.get_trainable_params()
print(f"Prompt Tuning trainable params: {pt_params:,}")  # ~50K

# Ratio
ratio = pt_params / mt_params * 100
print(f"Ratio: {ratio:.2f}%")  # ~0.02% - промпт туинг использует 500x меньше параметров!


# ============================================================================
# ПРИМЕР 8: Custom обработка результатов
# ============================================================================

import json
from pathlib import Path

# Загружаем результаты
results_file = Path("./results/custom_qa_model_tuning/results.json")
with open(results_file) as f:
    results = json.load(f)

# Метрики по датасетам
metrics = results["metrics"]
for dataset_name, dataset_metrics in metrics.items():
    print(f"\n{dataset_name}:")
    for metric_name, values in dataset_metrics.items():
        avg_value = np.mean(values)
        print(f"  {metric_name}: {avg_value:.4f}")

# Анализ переобучения
overfitting = results["overfitting_analysis"]
for metric_name, gap_value in overfitting.items():
    print(f"Overfitting {metric_name}: {gap_value:.4f}")

# Количество trainable параметров
trainable = results["trainable_params"]
print(f"\nTrainable parameters: {trainable:,}")


# ============================================================================
# ПРИМЕР 9: Batch запуск нескольких конфигураций
# ============================================================================

def run_batch_experiments():
    """Запускает серию экспериментов с разными гиперпараметрами"""
    
    configs = [
        ExperimentConfig(
            experiment_name="qa_mt_lr_1e3",
            task_type="qa",
            model_name="t5-base",
            tuning_approach="model_tuning",
            train_dataset="squad",
            eval_datasets=["squad_dev", "mrqa_newsqa"],
            learning_rate=1e-3,
        ),
        ExperimentConfig(
            experiment_name="qa_mt_lr_5e4",
            task_type="qa",
            model_name="t5-base",
            tuning_approach="model_tuning",
            train_dataset="squad",
            eval_datasets=["squad_dev", "mrqa_newsqa"],
            learning_rate=5e-4,
        ),
        ExperimentConfig(
            experiment_name="qa_pt_prompt_len_10",
            task_type="qa",
            model_name="t5-base",
            tuning_approach="prompt_tuning",
            train_dataset="squad",
            eval_datasets=["squad_dev", "mrqa_newsqa"],
            prompt_length=10,
        ),
        ExperimentConfig(
            experiment_name="qa_pt_prompt_len_30",
            task_type="qa",
            model_name="t5-base",
            tuning_approach="prompt_tuning",
            train_dataset="squad",
            eval_datasets=["squad_dev", "mrqa_newsqa"],
            prompt_length=30,
        ),
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config.experiment_name}")
        print(f"{'='*60}")
        
        experiment = QAExperiment(config)
        results = experiment.run()


# ============================================================================
# EXAMPLE 10: Troubleshooting guide
# ============================================================================

"""
ПРОБЛЕМА 1: CUDA Out of Memory
РЕШЕНИЕ:
    config.batch_size = 16  # Уменьшить batch size
    config.max_source_length = 256  # Уменьшить длину input
    
ПРОБЛЕМА 2: Низкие метрики
РЕШЕНИЕ:
    - Проверьте learning_rate (попробуйте 5e-4, 1e-4)
    - Для prompt tuning: увеличьте prompt_length (20, 30, 50)
    - Увеличьте количество epochs
    
ПРОБЛЕМА 3: Медленное обучение
РЕШЕНИЕ:
    - Используйте меньший модель (t5-small вместо t5-base)
    - Увеличьте batch_size (если VRAM позволяет)
    - Используйте GPU
    
ПРОБЛЕМА 4: Результаты несогласованны между runs
РЕШЕНИЕ:
    - Установите seed=42
    - Убедитесь, что детерминизм включен
    - Запустите несколько раз и усредните результаты
    
ПРОБЛЕМА 5: Prompt Tuning работает хуже, чем Model Tuning
РЕШЕНИЕ:
    - Увеличьте prompt_length (попробуйте 50-100)
    - Измените init_text на более информативный
    - Используйте более высокий learning_rate (1e-2)
    - Проверьте, что LM действительно заморожена
"""

# ============================================================================
# Рекомендации для публикации результатов
# ============================================================================

"""
РЕКОМЕНДАЦИИ:

1. Таблица сравнения:
   +-------------------+------------------+------------------+
   | Metric            | Model Tuning     | Prompt Tuning    |
   +-------------------+------------------+------------------+
   | Trainable Params  | 222M (100%)      | 50K (0.02%)      |
   | SQuAD F1          | 0.92             | 0.88             |
   | MRQA NewsQA F1    | 0.65 (-27%)      | 0.82 (-7%)       |
   | MRQA TriviaQA F1  | 0.68 (-26%)      | 0.80 (-9%)       |
   | Train-Test Gap    | 0.27             | 0.08             |
   +-------------------+------------------+------------------+

2. Графики:
   - Distribution shift по MRQA датасетам
   - Train vs Test curves для обеих подходов
   - Parameter count vs Performance trade-off

3. Анализ:
   - Attention patterns для обнаружения spurious correlations
   - Gradient flow для понимания обучения
   - Embedding change для анализа representation shift

4. Выводы:
   - Prompt tuning значительно лучше обобщается
   - Меньше параметров -> быстрее, эффективнее
   - Меньше риск переобучения -> более надежен на новых данных
"""
