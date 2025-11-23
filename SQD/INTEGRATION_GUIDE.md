# ИНТЕГРАЦИЯ В СУЩЕСТВУЮЩИЙ РЕПОЗИТОРИЙ

## Обзор

Эти файлы добавляют **раздел 5** для проверки гипотезы о переобучении при model tuning vs prompt tuning.

## Файлы для добавления

```
your_repo/
└── experiments/
    └── section5/
        ├── __init__.py                      (новый)
        ├── experiment_config.yaml           (новый)
        ├── experiments_section5.py          (основной модуль)
        ├── analysis_overfitting.py          (анализ результатов)
        ├── run_section5.py                  (entry point)
        ├── SECTION5_README.md               (документация)
        └── EXAMPLES_AND_QUICK_START.py     (примеры)
```

## Шаги интеграции

### 1. Копирование файлов

```bash
# В корне вашего репозитория
cd your_repo/experiments/
mkdir -p section5
cd section5

# Скопируйте все файлы из выше
cp /path/to/experiment_config.yaml .
cp /path/to/experiments_section5.py .
cp /path/to/analysis_overfitting.py .
cp /path/to/run_section5.py .
cp /path/to/SECTION5_README.md .
cp /path/to/EXAMPLES_AND_QUICK_START.py .
```

### 2. Создание __init__.py

```python
# experiments/section5/__init__.py

from .experiments_section5 import (
    ExperimentConfig,
    PromptTuningLayer,
    ModelTuningWrapper,
    PromptTuningWrapper,
    QAExperiment,
    ParaphraseExperiment,
    ExperimentTracker,
)

from .analysis_overfitting import OverfittingAnalyzer

__all__ = [
    "ExperimentConfig",
    "PromptTuningLayer",
    "ModelTuningWrapper",
    "PromptTuningWrapper",
    "QAExperiment",
    "ParaphraseExperiment",
    "ExperimentTracker",
    "OverfittingAnalyzer",
]
```

### 3. Обновление зависимостей

```bash
# Добавьте в requirements.txt вашего проекта
transformers>=4.20.0
datasets>=2.5.0
evaluate>=0.3.0
torch>=1.12.0
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.4.0
```

Или установите вручную:
```bash
pip install transformers datasets evaluate matplotlib seaborn pandas
```

## Использование в вашем коде

### Вариант 1: Standalone запуск

```bash
cd experiments/section5/
python run_section5.py
python run_section5.py --qa-only
python run_section5.py --paraphrase-only
python run_section5.py --no-experiments  # только анализ
```

### Вариант 2: Импорт в ваш pipeline

```python
# В вашем main training script
import sys
from pathlib import Path

# Добавьте section5 в path
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "section5"))

from experiments_section5 import QAExperiment, ExperimentConfig
from analysis_overfitting import OverfittingAnalyzer

# Используйте в своих скриптах
config = ExperimentConfig(...)
experiment = QAExperiment(config)
results = experiment.run()
```

### Вариант 3: Как часть CI/CD pipeline

```yaml
# .github/workflows/test_section5.yml
name: Section 5 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Section 5 experiments
        run: |
          cd experiments/section5/
          python run_section5.py --qa-only  # быстрый тест
      
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: section5-results
          path: experiments/section5/results/
```

## Интеграция с существующими экспериментами

### Если у вас уже есть base class для экспериментов

```python
# experiments/base_experiment.py (ваш существующий файл)

class BaseExperiment:
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def run(self):
        raise NotImplementedError
    
    def save_results(self):
        pass

# Добавьте наследование в section5
from experiments.base_experiment import BaseExperiment

class QAExperiment(BaseExperiment):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        # ... остальной код
```

### Если у вас есть базовая loss function

```python
# experiments/section5/experiments_section5.py

# Замените дефолтную loss на вашу
from your_repo.losses import CustomSeq2SeqLoss

class QAExperiment:
    def __init__(self, config):
        # ...
        self.loss_fn = CustomSeq2SeqLoss()  # используйте вашу
```

### Если у вас есть custom tokenizer

```python
# Используйте ваш custom tokenizer вместо AutoTokenizer
from your_repo.tokenizers import YourCustomTokenizer

class QAExperiment:
    def __init__(self, config):
        self.tokenizer = YourCustomTokenizer.from_pretrained(config.model_name)
```

## Расширение функциональности

### 1. Добавление новой метрики для переобучения

```python
# В analysis_overfitting.py добавьте метод

class OverfittingAnalyzer:
    def compute_my_custom_metric(self, experiment_name: str) -> float:
        """Ваша custom метрика"""
        metrics = self.results[experiment_name]["metrics"]
        # ... ваша логика
        return custom_value
```

### 2. Добавление новой задачи (например, Machine Translation)

```python
# В experiments_section5.py добавьте

class MTExperiment(BaseExperimentClass):
    """Machine Translation experiment"""
    
    def prepare_dataset(self, dataset_name: str, split: str = "train"):
        # Ваша логика для MT датасета
        pass
    
    def run(self):
        # Ваша логика для MT обучения
        pass
```

### 3. Добавление поддержки других моделей

```python
# В ExperimentConfig добавьте валидацию

@dataclass
class ExperimentConfig:
    model_name: str  # текущее
    
    def __post_init__(self):
        # Валидируем поддерживаемые модели
        supported = ["t5-small", "t5-base", "t5-large", "bart-base"]
        if self.model_name not in supported:
            raise ValueError(f"Model {self.model_name} not supported")
```

## Отладка и диагностика

### Проверка интеграции

```python
# test_integration.py

from experiments.section5 import (
    ExperimentConfig, QAExperiment, OverfittingAnalyzer
)

def test_integration():
    """Проверяет, что section5 хорошо интегрирован"""
    
    # 1. Проверяем импорты
    assert ExperimentConfig is not None
    assert QAExperiment is not None
    assert OverfittingAnalyzer is not None
    
    # 2. Проверяем создание конфига
    config = ExperimentConfig(
        experiment_name="test",
        task_type="qa",
        model_name="t5-base",
        tuning_approach="model_tuning",
        train_dataset="squad",
        eval_datasets=["squad_dev"],
    )
    assert config is not None
    
    # 3. Проверяем инициализацию эксперимента
    # (Не запускаем, только инициализируем)
    try:
        exp = QAExperiment(config)
        print("✓ Experiment initialization successful")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # 4. Проверяем анализ
    analyzer = OverfittingAnalyzer()
    assert analyzer is not None
    
    print("✓ All integration checks passed!")
    return True

if __name__ == "__main__":
    test_integration()
```

### Запуск тестов

```bash
# Добавьте в вашу CI/CD
python test_integration.py
```

## Обновление документации

### Добавить в основной README.md

```markdown
## Section 5: Overfitting Hypothesis

Experiments comparing Model Tuning vs Prompt Tuning generalization:

- **Experiment 1**: Question Answering (SQuAD + MRQA 2019)
- **Experiment 2**: Paraphrase Detection (QQP + MRPC)

See [experiments/section5/SECTION5_README.md](experiments/section5/SECTION5_README.md) for details.

### Quick Start
```bash
cd experiments/section5/
python run_section5.py
```
```

### Обновить Table of Contents

```markdown
# Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Experiments](#experiments)
   - [Section 1: Basics](#section-1)
   - [Section 2: Advanced](#section-2)
   - ...
   - **[Section 5: Overfitting Hypothesis](#section-5)** ← NEW
5. [Results](#results)
6. [Contributing](#contributing)
```

## Типичный workflow

```bash
# 1. Сделали изменения -> запустили тесты
python experiments/section5/run_section5.py --qa-only

# 2. Результаты хорошие -> коммитим
git add experiments/section5/
git commit -m "feat: add section5 overfitting experiments"

# 3. Пушим в main branch
git push origin main

# 4. CI/CD запускает полные тесты
# (GitHub Actions, GitLab CI, и т.д.)

# 5. Результаты автоматически сохраняются
# (в artifacts, в S3, и т.д.)
```

## FAQ по интеграции

**Q: Где хранятся результаты?**
A: В `./results/` (или путь, указанный в config)

**Q: Можно ли использовать GPU?**
A: Да, автоматически используется GPU если доступен

**Q: Как изменить базовый путь для сохранения результатов?**
A: 
```python
config = ExperimentConfig(..., output_dir="/path/to/results")
```

**Q: Можно ли запустить параллельно несколько экспериментов?**
A: Да, используйте `multiprocessing` или запустите на разных машинах

**Q: Как получить только метрики без полного сохранения модели?**
A: Используйте `--no-experiments` флаг и параметр `save_total_limit=1`

**Q: Совместимо ли с DDP (Distributed Data Parallel)?**
A: Да, `Seq2SeqTrainer` поддерживает DDP из коробки

## Контакт и поддержка

Для вопросов по интеграции:
1. Проверьте SECTION5_README.md
2. Посмотрите EXAMPLES_AND_QUICK_START.py
3. Откройте issue на GitHub
