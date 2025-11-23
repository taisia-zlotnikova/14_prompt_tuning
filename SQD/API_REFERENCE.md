# РАЗДЕЛ 5: API REFERENCE

## 🔧 Полная документация всех классов и методов

---

## 1. ExperimentConfig (dataclass)

**Назначение**: Конфигурация эксперимента

```python
from experiments_section5 import ExperimentConfig

config = ExperimentConfig(
    experiment_name: str,           # Название эксперимента
    task_type: str,                 # "qa" или "paraphrase"
    model_name: str,                # "t5-base", "t5-large", и т.д.
    tuning_approach: str,           # "model_tuning" или "prompt_tuning"
    
    # Dataset configuration
    train_dataset: str,             # "squad", "qqp", "mrpc"
    eval_datasets: List[str],       # ["squad_dev", "mrqa_newsqa", ...]
    
    # Model hyperparameters
    max_source_length: int = 512,   # Максимальная длина входа
    max_target_length: int = 50,    # Максимальная длина выхода
    batch_size: int = 32,           # Размер батча
    num_epochs: int = 3,            # Количество эпох
    learning_rate: float = 1e-3,    # Learning rate
    warmup_steps: int = 500,        # Warmup steps
    
    # Prompt tuning specific
    prompt_length: int = 20,        # Длина soft prompt
    init_text: str = "",            # Инициализация из vocabulary
    
    # Output
    output_dir: str = "./results",  # Директория для результатов
    seed: int = 42,                 # Random seed
)
```

**Примеры использования**:

```python
# QA experiment with model tuning
config_mt = ExperimentConfig(
    experiment_name="qa_mt",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="model_tuning",
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"],
    learning_rate=1e-3,
)

# QA experiment with prompt tuning
config_pt = ExperimentConfig(
    experiment_name="qa_pt",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="prompt_tuning",
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"],
    prompt_length=20,
    init_text="question answering: ",
    learning_rate=1e-3,
)

# Paraphrase experiment
config_para = ExperimentConfig(
    experiment_name="paraphrase_mt",
    task_type="paraphrase",
    model_name="t5-base",
    tuning_approach="model_tuning",
    train_dataset="qqp",
    eval_datasets=["qqp", "mrpc"],
    num_epochs=10,
)
```

---

## 2. PromptTuningLayer

**Назначение**: Реализует обучаемые soft prompts (мягкие подсказки)

```python
from experiments_section5 import PromptTuningLayer

layer = PromptTuningLayer(
    prompt_length: int,     # Длина промпта в токенах
    hidden_size: int,       # Размер скрытого слоя модели
    vocab_size: int = None  # Опциональный размер словаря
)
```

**Методы**:

```python
def forward(self, encoder_outputs, attention_mask=None):
    """
    Конкатенирует soft prompts с выходом encoder'а
    
    Args:
        encoder_outputs: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, seq_len)
    
    Returns:
        extended_outputs: (batch_size, prompt_len + seq_len, hidden_size)
        extended_mask: (batch_size, prompt_len + seq_len)
    """
```

**Как работает**:

1. Инициализирует trainable параметр `soft_prompt` размером `(1, prompt_length, hidden_size)`
2. Расширяет его для всего batch'а
3. Конкатенирует с `encoder_outputs`: `[soft_prompt; encoder_hidden_states]`
4. Расширяет `attention_mask` соответственно

---

## 3. ModelTuningWrapper

**Назначение**: Оборачивает модель для model tuning (обновляются ВСЕ параметры)

```python
from experiments_section5 import ModelTuningWrapper
from transformers import AutoModelForSeq2SeqLM

base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
wrapped_model = ModelTuningWrapper(base_model)
```

**Методы**:

```python
def forward(self, **kwargs):
    """Просто пропускает через модель"""
    return self.model(**kwargs)

def get_trainable_params(self) -> int:
    """Возвращает количество trainable параметров"""
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
```

**Конфигурация**:

- ✓ ВСЕ параметры: `requires_grad = True`
- ✓ Никакого замораживания

**Результат**:

- ~220M trainable параметров для t5-base
- Полное обновление при backprop
- Высокий риск переобучения

---

## 4. PromptTuningWrapper

**Назначение**: Оборачивает модель для prompt tuning (заморожены параметры LM)

```python
from experiments_section5 import PromptTuningWrapper
from transformers import AutoModelForSeq2SeqLM

base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
wrapped_model = PromptTuningWrapper(base_model, prompt_length=20)
```

**Методы**:

```python
def forward(self, 
    input_ids=None, 
    attention_mask=None,
    decoder_input_ids=None, 
    labels=None, 
    **kwargs
):
    """
    Forward pass с soft prompts
    
    Returns:
        dict with:
        - loss: scalar loss
        - logits: (batch_size, seq_len, vocab_size)
    """

def get_trainable_params(self) -> int:
    """Возвращает количество trainable параметров (только prompts)"""
```

**Конфигурация**:

- ✓ LM параметры: `requires_grad = False`
- ✓ Prompt параметры: `requires_grad = True`

**Результат**:

- ~50K trainable параметров для t5-base + prompt_length=20
- Только prompts обновляются при backprop
- Низкий риск переобучения

---

## 5. ExperimentTracker

**Назначение**: Отслеживает метрики во время экспериментов

```python
from experiments_section5 import ExperimentTracker

tracker = ExperimentTracker(experiment_name="my_experiment")
```

**Методы**:

```python
def record(self, dataset_name: str, metric_name: str, value: float):
    """
    Записывает метрику
    
    Args:
        dataset_name: название датасета ("squad_dev", "mrqa_newsqa", и т.д.)
        metric_name: название метрики ("f1", "exact_match", и т.д.)
        value: значение метрики (float)
    """

def compute_overfitting_metrics(self,
    train_dataset: str,
    eval_datasets: List[str]
) -> Dict[str, float]:
    """
    Вычисляет метрики переобучения (train-test gaps)
    
    Returns:
        dict: {
            "eval_dataset_metric_gap": gap_value,
            ...
        }
    """

def to_dict(self) -> Dict:
    """Экспортирует все метрики в словарь"""
```

---

## 6. QAExperiment

**Назначение**: Эксперимент 1 - Question Answering (SQuAD + MRQA)

```python
from experiments_section5 import QAExperiment, ExperimentConfig

config = ExperimentConfig(
    experiment_name="qa_experiment",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="model_tuning",
    train_dataset="squad",
    eval_datasets=["squad_dev", "mrqa_newsqa"],
)

exp = QAExperiment(config)
results = exp.run()
```

**Методы**:

```python
def prepare_dataset(self, dataset_name: str, split: str = "train"):
    """
    Загружает и препроцессирует датасет QA
    
    Args:
        dataset_name: "squad", "squad_dev", "mrqa_newsqa", и т.д.
        split: "train" или "validation"
    
    Returns:
        HF Dataset с полями: input_ids, attention_mask, labels
    """

def run(self) -> Dict:
    """
    Запускает полный эксперимент: обучение + оценка
    
    Returns:
        dict: {
            "config": {...},
            "metrics": {...},
            "overfitting_analysis": {...},
            "trainable_params": int
        }
    """
```

**Датасеты**:

- **Train**: SQuAD (88K примеров)
- **Eval**:
  - In-distribution: SQuAD validation
  - Out-of-distribution: MRQA 2019 (NewsQA, TriviaQA, SearchQA, HotpotQA)

**Метрики**: Exact Match, F1

---

## 7. ParaphraseExperiment

**Назначение**: Эксперимент 2 - Paraphrase Detection (QQP + MRPC)

```python
from experiments_section5 import ParaphraseExperiment, ExperimentConfig

config = ExperimentConfig(
    experiment_name="paraphrase_experiment",
    task_type="paraphrase",
    model_name="t5-base",
    tuning_approach="model_tuning",
    train_dataset="qqp",
    eval_datasets=["qqp", "mrpc"],
    num_epochs=10,
)

exp = ParaphraseExperiment(config)
results = exp.run_cross_dataset_eval()
```

**Методы**:

```python
def prepare_dataset(self, dataset_name: str, split: str = "train"):
    """
    Загружает и препроцессирует датасет Paraphrase
    
    Args:
        dataset_name: "qqp" или "mrpc"
        split: "train" или "validation"
    
    Returns:
        HF Dataset с полями: input_ids, attention_mask, labels
    """

def run_cross_dataset_eval(self) -> Dict:
    """
    Запускает кросс-датасетное обучение и оценку
    
    Обучает модель на одном датасете и тестирует на обоих:
    - Train QQP → Test QQP + MRPC
    - Train MRPC → Test MRPC + QQP
    
    Returns:
        dict: результаты всех комбинаций
    """

def _analyze_distribution_shift(self) -> Dict[str, float]:
    """Анализирует влияние distribution shift"""
```

**Датасеты**:

- QQP: Quora Question Pairs (~364K примеров)
- MRPC: Microsoft Research Paraphrase Corpus (~5.7K примеров)

**Метрики**: Accuracy, F1

---

## 8. OverfittingAnalyzer

**Назначение**: Анализирует результаты экспериментов и переобучение

```python
from analysis_overfitting import OverfittingAnalyzer

analyzer = OverfittingAnalyzer(results_dir="./results")
```

**Методы**:

```python
def load_results(self, experiment_name: str):
    """Загружает результаты эксперимента из JSON"""

def compute_train_test_gap(self,
    experiment_name: str,
    train_dataset: str,
    eval_datasets: List[str]
) -> Dict[str, float]:
    """
    Вычисляет gap между train и eval метриками
    
    Gap = |Train_Metric - Eval_Metric|
    
    Returns:
        dict: {"eval_dataset_metric_gap": gap_value, ...}
    """

def compute_distribution_shift_impact(self,
    experiment_name: str,
    in_dist_dataset: str,
    ood_datasets: List[str]
) -> Dict[str, float]:
    """
    Вычисляет падение производительности на OOD данных
    
    Drop% = (InDist - OOD) / InDist * 100
    
    Returns:
        dict: {"ood_dataset_metric_drop_pct": drop_pct, ...}
    """

def compute_spurious_correlation_proxy(self,
    experiment_name: str
) -> Dict[str, float]:
    """
    Анализирует ложные корреляции через коэффициент вариации
    
    CV = StdDev(scores_across_datasets) / Mean(scores)
    
    High CV -> dataset-specific overfitting
    Low CV -> general features
    
    Returns:
        dict: {"metric_name_cv": cv_value, ...}
    """

def compare_tuning_approaches(self,
    model_tuning_exp: str,
    prompt_tuning_exp: str
) -> pd.DataFrame:
    """
    Сравнивает Model Tuning vs Prompt Tuning
    
    Returns:
        DataFrame с метриками для обоих подходов
    """

def visualize_distribution_shift(self,
    experiment_name: str,
    output_file: str = None):
    """
    Визуализирует падение производительности на OOD данных
    
    Создает графики для каждой метрики с in-dist и OOD результатами
    """

def generate_report(self,
    model_tuning_exp: str,
    prompt_tuning_exp: str,
    output_file: str = "report.txt"):
    """
    Генерирует полный текстовый отчет
    
    Содержит:
    - Гипотезу
    - Сравнительные метрики
    - Интерпретацию
    - Вывод
    """
```

---

## 9. Функции для запуска

```python
# run_section5.py

def run_qa_experiments() -> Tuple[str, str]:
    """Запускает QA эксперименты (MT + PT), возвращает имена экспериментов"""

def run_paraphrase_experiments() -> Tuple[str, str]:
    """Запускает Paraphrase эксперименты (MT + PT), возвращает имена экспериментов"""

def analyze_results(qa_mt_exp: str, qa_pt_exp: str,
                   para_mt_exp: str, para_pt_exp: str):
    """Анализирует результаты всех экспериментов"""

def main():
    """Main entry point с парсингом аргументов"""
```

**Аргументы командной строки**:

```bash
python run_section5.py --qa-only              # Только QA
python run_section5.py --paraphrase-only      # Только Paraphrase
python run_section5.py --no-experiments       # Только анализ
python run_section5.py                        # Все
```

---

## 10. Типичные комбинации использования

### Scenario 1: Quick test

```python
from experiments_section5 import QAExperiment, ExperimentConfig

config = ExperimentConfig(
    experiment_name="quick_test",
    task_type="qa",
    model_name="t5-base",
    tuning_approach="model_tuning",
    train_dataset="squad",
    eval_datasets=["squad_dev"],
    num_epochs=1,  # Только 1 эпоха для быстрого теста
)

exp = QAExperiment(config)
results = exp.run()
```

### Scenario 2: Full comparison

```python
# Запустить оба подхода
configs = [
    ExperimentConfig(..., tuning_approach="model_tuning"),
    ExperimentConfig(..., tuning_approach="prompt_tuning"),
]

for config in configs:
    exp = QAExperiment(config)
    results = exp.run()

# Проанализировать
analyzer = OverfittingAnalyzer()
analyzer.compare_tuning_approaches(
    model_tuning_exp="qa_model_tuning",
    prompt_tuning_exp="qa_prompt_tuning"
)
analyzer.generate_report(...)
```

### Scenario 3: Custom metrics

```python
# Загружаем результаты
with open("results/qa_model_tuning/results.json") as f:
    results = json.load(f)

# Извлекаем метрики
metrics = results["metrics"]
for dataset, scores in metrics.items():
    avg_f1 = np.mean(scores["f1"])
    print(f"{dataset}: F1={avg_f1:.4f}")
```

---

## 11. Структура результатов (results.json)

```json
{
  "config": {
    "experiment_name": "qa_model_tuning",
    "task_type": "qa",
    "model_name": "t5-base",
    "tuning_approach": "model_tuning",
    "train_dataset": "squad",
    "eval_datasets": ["squad_dev", "mrqa_newsqa"],
    ...
  },
  "metrics": {
    "squad_dev": {
      "exact_match": [0.80, 0.81, 0.82],
      "f1": [0.85, 0.86, 0.87]
    },
    "mrqa_newsqa": {
      "exact_match": [0.60, 0.61, 0.62],
      "f1": [0.65, 0.66, 0.67]
    }
  },
  "overfitting_analysis": {
    "squad_dev_f1_gap": 0.05,
    "mrqa_newsqa_f1_gap": 0.20
  },
  "trainable_params": 222000000
}
```

---

## 12. Расширение функциональности

### Добавить новый датасет

```python
# В prepare_dataset методе
if dataset_name == "my_dataset":
    dataset = load_dataset("my_dataset")
    # ... препроцессинг
```

### Добавить новую метрику

```python
# В OverfittingAnalyzer
def compute_my_metric(self, experiment_name: str) -> Dict:
    metrics = self.results[experiment_name]["metrics"]
    # ... ваша логика
    return custom_metric
```

### Добавить новую задачу

```python
# Создать новый класс
class MyExperiment(BaseExperimentClass):
    def prepare_dataset(self, ...):
        pass
    
    def run(self):
        pass
```

---

**Version**: 1.0  
**Status**: Complete API Reference  
**Last Updated**: November 2025
