# Section 5: Overfitting Hypothesis - Model Tuning vs Prompt Tuning

Интеграция экспериментов для проверки гипотезы о переобучении (Раздел 5)

## Обзор

Этот раздел добавляет два крупномасштабных эксперимента:

### Эксперимент 1: Question Answering (QA) на MRQA 2019
- **Датасеты**: SQuAD (train) + SQuAD (test) + MRQA 2019 OOD (newsqa, triviaqa, searchqa, hotpotqa)
- **Задача**: Измерить, как хорошо модель обобщается на датасеты с другим распределением
- **Гипотеза**: Model tuning переобучится на SQuAD, prompt tuning обобщается лучше

### Эксперимент 2: Paraphrase Detection (QQP + MRPC)
- **Датасеты**: 
  - Train on QQP, eval on QQP + MRPC
  - Train on MRPC, eval on MRPC + QQP
- **Задача**: Кросс-датасетное тестирование на задаче парафразирования
- **Гипотеза**: Prompt tuning менее восприимчив к spurious correlations в QQP/MRPC

## Структура проекта

```
your_repo/
├── experiments/
│   ├── section5/
│   │   ├── experiment_config.yaml          # Конфигурация экспериментов
│   │   ├── experiments_section5.py         # Основной код экспериментов
│   │   │   ├── ExperimentConfig
│   │   │   ├── PromptTuningLayer          # Soft prompts (обучаемые префиксы)
│   │   │   ├── ModelTuningWrapper          # Full parameter tuning
│   │   │   ├── PromptTuningWrapper         # Frozen LM + learnable prompts
│   │   │   ├── QAExperiment               # Impl. Experiment 1
│   │   │   └── ParaphraseExperiment       # Impl. Experiment 2
│   │   ├── analysis_overfitting.py        # Анализ и визуализация
│   │   │   └── OverfittingAnalyzer
│   │   │       ├── compute_train_test_gap
│   │   │       ├── compute_distribution_shift_impact
│   │   │       ├── compute_spurious_correlation_proxy
│   │   │       └── generate_report
│   │   └── run_section5.py               # Main entry point
│   │
│   └── results/
│       ├── qa_model_tuning/
│       │   └── results.json
│       ├── qa_prompt_tuning/
│       │   └── results.json
│       ├── paraphrase_model_tuning/
│       │   └── results.json
│       ├── paraphrase_prompt_tuning/
│       │   └── results.json
│       ├── qa_overfitting_report.txt
│       ├── paraphrase_overfitting_report.txt
│       └── section5_summary_report.txt
```

## Установка и запуск

### 1. Установка зависимостей

```bash
pip install transformers datasets evaluate torch
pip install matplotlib seaborn pandas  # для анализа
```

### 2. Запуск всех экспериментов

```bash
cd your_repo/experiments/section5/
python run_section5.py
```

### 3. Запуск отдельных экспериментов

```bash
# Только QA эксперименты
python run_section5.py --qa-only

# Только Paraphrase эксперименты
python run_section5.py --paraphrase-only

# Анализ существующих результатов (без переобучения)
python run_section5.py --no-experiments
```

## Метрики для оценки переобучения

### 1. Train-Test Gap
```
Gap = |Train Metric - Test Metric|
Интерпретация:
  - Высокий gap = модель переобучилась
  - Низкий gap = лучшая обобщаемость
```

### 2. Distribution Shift Impact
```
Drop% = (InDist_Perf - OOD_Perf) / InDist_Perf * 100
Интерпретация:
  - Низкий drop = устойчивость к distribution shift
  - Высокий drop = модель запомнила InDist特иности
```

### 3. Spurious Correlation Proxy (Coefficient of Variation)
```
CV = StdDev(scores_across_datasets) / Mean(scores_across_datasets)
Интерпретация:
  - Высокий CV = разные датасеты дают разные scores
            -> модель выучила dataset-specific features
  - Низкий CV = consistent performance
           -> модель выучила general features
```

## Ожидаемые результаты

Если гипотеза верна:

| Метрика | Model Tuning | Prompt Tuning | Ожидание |
|---------|-------------|---------------|----------|
| Train-Test Gap | Высокий | Низкий | ✓ |
| Distribution Shift Drop | Высокий | Низкий | ✓ |
| CV (Spurious Corr.) | Высокий | Низкий | ✓ |
| Trainable Parameters | ~100% | ~0.5-1% | ✓ |

## Интеграция в существующий репозиторий

### Шаг 1: Добавить в основной тренировочный pipeline

```python
from experiments.section5.experiments_section5 import QAExperiment, ExperimentConfig

# В вашем main training script
config = ExperimentConfig(
    experiment_name="my_qa_experiment",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="prompt_tuning",  # или "model_tuning"
    train_dataset="squad",
    eval_datasets=["mrqa_newsqa", "mrqa_triviaqa"],
)

experiment = QAExperiment(config)
results = experiment.run()
```

### Шаг 2: Анализ результатов

```python
from experiments.section5.analysis_overfitting import OverfittingAnalyzer

analyzer = OverfittingAnalyzer(results_dir="./results")
analyzer.generate_report(
    model_tuning_exp="qa_model_tuning",
    prompt_tuning_exp="qa_prompt_tuning",
    output_file="report.txt"
)
```

### Шаг 3: Визуализация

```python
analyzer.visualize_distribution_shift(
    experiment_name="qa_model_tuning",
    output_file="distribution_shift.png"
)
```

## Дополнительные возможности расширения

### 1. Attention Analysis для детекции spurious correlations
```python
# Извлечь attention patterns
attention_scores = model.encoder.layer[0].attention.self.attention_probs
# Визуализировать, какие токены моделью используются для prediction
```

### 2. Feature Importance Analysis
```python
# Gradient-based: какие input features наиболее влияют на prediction
# Усредненные градиенты по dataset - показывают spurious correlations
```

### 3. Vocabulary Specialization
```python
# Анализ embedding специализации при model tuning
# Embedding divergence между prompts в PT vs заученных параметров в MT
```

### 4. Fine-grained Distribution Shift Analysis
```python
# Анализ accuracy в зависимости от:
#   - Длины контекста
#   - Типа вопроса
#   - Лексической сложности
```

## Параметры для экспериментов

### QA Configuration
```yaml
model: t5-base  # или t5-small для быстрых экспериментов
batch_size: 32
epochs: 3
learning_rate: 1e-3
warmup_steps: 500
prompt_length: 20  # для prompt tuning
```

### Paraphrase Configuration
```yaml
model: t5-base
batch_size: 32
epochs: 10  # больше эпох для паraffrazирования
learning_rate: 1e-3
warmup_steps: 500
prompt_length: 15  # меньше для более простой задачи
```

## Диагностика проблем

### CUDA Out of Memory
```python
# Уменьшите batch_size в ExperimentConfig
config.batch_size = 16  # или 8
```

### Медленное скачивание датасетов
```python
# Датасеты кешируются в ~/.cache/huggingface/
# Удалите для переоткачивания: rm -rf ~/.cache/huggingface/
```

### Низкие метрики
```python
# Проверьте:
# 1. learning_rate - попробуйте 1e-4 или 5e-4
# 2. prompt_length - попробуйте 10, 15, 30
# 3. init_text - проверьте качество инициализации
```

## Цитирование

Если используете эти эксперименты, укажите:

```
@section{section5_overfitting_2024,
  title={Overfitting Hypothesis: Model Tuning vs Prompt Tuning},
  author={Your Lab},
  year={2024},
  url={https://github.com/...}
}
```

## Заметки

1. **Reproducibility**: Используйте `seed=42` для воспроизводимости
2. **GPU требования**: ~10GB VRAM для t5-base с batch_size=32
3. **Время обучения**: ~2-3 часа для QA, ~1 час для Paraphrase на одном V100
4. **Холодный старт**: Первый запуск медленнее (скачивание моделей и датасетов)

## Контакт

Для вопросов по интеграции или модификации экспериментов см. разработку в вашем репозитории.
