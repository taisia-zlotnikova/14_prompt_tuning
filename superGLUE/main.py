import torch
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
)
import evaluate

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    MODEL_NAME = "t5-small"  # или "google/t5-base"
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 10
    NUM_VIRTUAL_TOKENS = 20  # Для prompt tuning
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

# ============================================================================
# ПОДГОТОВКА ДАННЫХ - SuperGLUE CB в T5 формат
# ============================================================================

class DataPreprocessor:
    """Загружает SuperGLUE CB и преобразует в T5 text-to-text формат"""

    def __init__(self, tokenizer, max_input_len=512, max_target_len=10):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.label_map = {0: "entailment", 1: "contradiction", 2: "neutral"}

    def load_superglue_cb(self):
        """Загружаем датасет"""
        return load_dataset("super_glue", "cb", trust_remote_code=True)

    def format_for_t5(self, example):
        """
        Преобразуем в T5 формат.

        SuperGLUE CB структура:
        - premise: основное предложение
        - hypothesis: встроенное предложение
        - label: 0 (entailment), 1 (contradiction), 2 (neutral)

        T5 text-to-text формат:
        Input: "premise: <premise> hypothesis: <hypothesis>"
        Output: "entailment" или "contradiction" или "neutral"
        """
        input_text = f"premise: {example['premise']} hypothesis: {example['hypothesis']}"
        target_text = self.label_map[example['label']]

        return {
            'input_text': input_text,
            'target_text': target_text,
            'label': example['label'],
        }

    def preprocess_function(self, examples):
        """Токенизация входов и целей"""
        inputs = [ex for ex in examples['input_text']]
        targets = [ex for ex in examples['target_text']]

        original_labels = examples['label']

        # Токенизируем входы
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Токенизируем целевые значения (ответы модели)
        labels = self.tokenizer(
            targets,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Для T5: labels = input_ids, заменяем padding на -100
        # (-100 в loss функции игнорируется)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': model_inputs['labels'],
            'label': original_labels,
        }

    def prepare_dataset(self):
        """Загружаем и подготавливаем датасет"""
        dataset = self.load_superglue_cb()

        print(dataset['train'].column_names)

        dataset = dataset.filter(lambda x: x['label'] in [0, 1, 2])

        # Применяем форматирование
        dataset = dataset.map(
            self.format_for_t5
        )

        print(dataset['train'].column_names)

        # Токенизируем
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
        )

        return tokenized_dataset

# ============================================================================
# ПОДХОД 1: FINE-TUNING
# ============================================================================

class FineTuningApproach:
    """
    ЧТО ОБУЧАЕМ: Все веса модели
    ЧТО ЗАМОРАЖИВАЕМ: Ничего

    Все параметры T5 обновляются градиентом.
    """

    def __init__(self, model_name: str, config: Config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(config.DEVICE)

        # ВСЕ параметры по умолчанию обучаемые (requires_grad=True)

    def prepare_trainer(self, train_dataset, eval_dataset):
        """Настраиваем обучение"""
        training_args = TrainingArguments(
            output_dir="./results/fine_tuning",
            learning_rate=self.config.LEARNING_RATE,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            num_train_epochs=self.config.NUM_EPOCHS,
            save_strategy="epoch",
            eval_strategy="epoch",
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        return trainer

    def count_parameters(self):
        """Считаем обучаемые параметры"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            'trainable': trainable,
            'total': total,
            'trainable_pct': 100 * trainable / total,
        }

# ============================================================================
# ПОДХОД 2: PROMPT DESIGN
# ============================================================================

class PromptDesignApproach:
    """
    ЧТО ОБУЧАЕМ: Ничего!
    ЧТО ЗАМОРАЖИВАЕМ: Все веса модели

    Ручно создаём текстовую подсказку.
    Это baseline из GPT-3 paper.
    """

    def __init__(self, model_name: str, config: Config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(config.DEVICE)

        # ЗАМОРАЖИВАЕМ все параметры
        for param in self.model.parameters():
            param.requires_grad = False

        # Ручно разработанные подсказки
        self.prompts = {
            "instruction": "Определи, следует ли гипотеза из предпосылки.",
            "few_shot": (
                "premise: Присяжные будут голосовать за закон. "
                "hypothesis: Присяжные будут голосовать. entailment\n"
            ),
        }

    def construct_prompt(self, premise: str, hypothesis: str):
        """Конструируем ручную подсказку для инференса"""
        prompt = (
            f"{self.prompts['instruction']}\n\n"
            f"{self.prompts['few_shot']}\n"
            f"premise: {premise} hypothesis: {hypothesis}"
        )
        return prompt

    def count_parameters(self):
        """Prompt design не имеет обучаемых параметров"""
        return {
            'trainable': 0,
            'total': sum(p.numel() for p in self.model.parameters()),
            'trainable_pct': 0,
            'prompt_tokens': len(self.tokenizer(self.prompts['instruction']).input_ids) +
                           len(self.tokenizer(self.prompts['few_shot']).input_ids),
        }

# ============================================================================
# ПОДХОД 3: PROMPT TUNING
# ============================================================================

class PromptTuningApproach:
    """
    ЧТО ОБУЧАЕМ: Только мягкие embeddings подсказок (20 × 512 = ~10K параметров)
    ЧТО ЗАМОРАЖИВАЕМ: ВСЕ веса T5 модели

    Ключевая идея:
    - Добавляем K обучаемых embedding векторов в начало входа
    - Это НЕ дискретные токены из словаря
    - Это continuous vectors размера [K, embedding_dim]
    - Только эти векторы обновляются через градиент
    """

    def __init__(self, model_name: str, config: Config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Загружаем базовую модель
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # КЛЮЧЕВОЙ МОМЕНТ: Конфигурируем prompt tuning
        peft_config = PromptTuningConfig(
            # Задача: последовательность в последовательность (генерация)
            task_type=TaskType.SEQ_2_SEQ_LM,

            # Как инициализировать мягкие подсказки
            prompt_tuning_init=PromptTuningInit.RANDOM,

            # ГЛАВНОЕ: Количество обучаемых токенов
            # К = 20 означает: добавляем 20 embedding vectors размером 512 каждый
            # Всего параметров: 20 * 512 = 10,240
            num_virtual_tokens=config.NUM_VIRTUAL_TOKENS,

            # Путь к токенайзеру (нужен для инициализации)
            tokenizer_name_or_path=model_name,
        )

        # Оборачиваем модель в PEFT (Parameter-Efficient Fine-Tuning)
        # Это добавляет слой мягких подсказок сверху T5
        self.model = get_peft_model(model, peft_config)
        self.model.to(config.DEVICE)

    def prepare_trainer(self, train_dataset, eval_dataset):
        """Настраиваем обучение (как в fine-tuning)"""
        training_args = TrainingArguments(
            output_dir="./results/prompt_tuning",
            learning_rate=self.config.LEARNING_RATE,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            num_train_epochs=self.config.NUM_EPOCHS,
            save_strategy="epoch",
            eval_strategy="epoch",
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        return trainer

    def count_parameters(self):
        """Считаем параметры: должны быть только мягкие подсказки"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            'trainable': trainable,
            'total': total,
            'trainable_pct': 100 * trainable / total,
        }


# ============================================================================
# ТЕСТИРОВАНИЕ МОДЕЛЕЙ
# ============================================================================


def evaluate_approach(approach, test_dataset, tokenizer, approach_name):
    """
    Оцениваем подход на test датасете

    Args:
        approach: объект подхода (FineTuning, PromptDesign, или PromptTuning)
        test_dataset: test датасет
        tokenizer: T5Tokenizer
        approach_name: название подхода для логирования
    """
    print("\n" + "=" * 80)
    print(f"ТЕСТИРОВАНИЕ: {approach_name.upper()}")
    print("=" * 80)

    # Переводим модель в режим оценки
    approach.model.eval()
    device = approach.config.DEVICE

    predictions = []
    true_labels = []

    # Инвертированный label_map для преобразования предсказаний в классы
    id_to_label = {0: "entailment", 1: "contradiction", 2: "neutral"}
    label_to_id = {v: k for k, v in id_to_label.items()}

    print(f"\nОценка на {len(test_dataset)} примерах...")

    # Инференс
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i % 10 == 0:
                print(f"  Обработано {i}/{len(test_dataset)} примеров...")

            # Подготавливаем вход
            input_ids = torch.tensor([example['input_ids']]).to(device)
            attention_mask = torch.tensor([example['attention_mask']]).to(device)

            # Генерируем предсказание
            outputs = approach.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=10,
                num_beams=1,
            )

            # Декодируем предсказание
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Получаем истинный label
            true_label_id = example['label']  # Первый token после padding

            predictions.append(predicted_text)
            true_labels.append(true_label_id)

    # Вычисляем метрики
    metrics = compute_metrics(predictions, true_labels, label_to_id)

    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({approach_name}):")
    print("-" * 80)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    # Показываем примеры предсказаний
    print(f"\n📌 Примеры предсказаний:")
    print("-" * 80)
    for i in range(min(5, len(test_dataset))):
        true_label_text = id_to_label[true_labels[i]]
        pred_label_text = predictions[i]
        is_correct = "✓" if true_label_text == pred_label_text else "✗"

        print(f"\nПример {i} {is_correct}:")
        print(f"  Истинный:      {true_label_text}")
        print(f"  Предсказание:  {pred_label_text}")

    return metrics


def compute_metrics(predictions, true_labels, label_to_id):
    """
    Вычисляем метрики качества

    Args:
        predictions: список строк с предсказаниями ("entailment", "contradiction", "neutral")
        true_labels: список истинных label IDs (0, 1, 2)
        label_to_id: словарь для преобразования текста в ID

    Returns:
        dict с метриками
    """
    id_to_label = {0: "entailment", 1: "contradiction", 2: "neutral"}

    # Преобразуем предсказания в IDs
    pred_ids = []
    for pred in predictions:
        # Ищем ближайший label
        pred_lower = pred.lower().strip()
        found = False

        # Точное совпадение
        for label_text, label_id in label_to_id.items():
            if pred_lower == label_text:
                pred_ids.append(label_id)
                found = True
                break

        # Если не найдено, ищем частичное совпадение
        if not found:
            for label_text, label_id in label_to_id.items():
                if label_text in pred_lower or pred_lower in label_text:
                    pred_ids.append(label_id)
                    found = True
                    break

        # Если совпадение не найдено, берём максимально вероятный класс
        if not found:
            pred_ids.append(0)  # По умолчанию entailment

    # Вычисляем метрики
    correct = sum(1 for p, t in zip(pred_ids, true_labels) if p == t)
    total = len(true_labels)

    # Accuracy
    accuracy = correct / total

    # Precision, Recall, F1 для каждого класса
    precision_per_class = []
    recall_per_class = []

    for class_id in [0, 1, 2]:
        tp = sum(1 for p, t in zip(pred_ids, true_labels) if p == t and t == class_id)
        fp = sum(1 for p, t in zip(pred_ids, true_labels) if p == class_id and t != class_id)
        fn = sum(1 for p, t in zip(pred_ids, true_labels) if p != class_id and t == class_id)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    # Макро-усреднение
    macro_precision = sum(precision_per_class) / len(precision_per_class)
    macro_recall = sum(recall_per_class) / len(recall_per_class)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'correct': correct,
        'total': total,
    }


def compare_all_approaches(approaches_dict, test_dataset, tokenizer):
    """
    Сравниваем все подходы на test датасете
    """
    print("\n" + "=" * 80)
    print("ФИНАЛЬНОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ НА TEST ДАТАСЕТЕ")
    print("=" * 80)

    results = {}

    for approach_name, approach in approaches_dict.items():
        metrics = evaluate_approach(approach, test_dataset, tokenizer, approach_name)
        results[approach_name] = metrics

    # Таблица сравнения
    print("\n" + "=" * 80)
    print("ТАБЛИЦА СРАВНЕНИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    print(f"\n{'Подход':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)

    for approach_name, metrics in results.items():
        print(f"{approach_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

    # Выводы
    print("\n" + "=" * 80)
    print("ВЫВОДЫ")
    print("=" * 80)

    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])

    print(f"\n✓ Лучший результат по Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']*100:.2f}%)")
    print(f"✓ Лучший результат по F1-Score:  {best_f1[0]} ({best_f1[1]['f1']:.4f})")

    return results

# ============================================================================
# ГЛАВНЫЙ КОД: СРАВНЕНИЕ ВСЕХ ТРЁХ ПОДХОДОВ
# ============================================================================

def main():
    config = Config()

    print("=" * 80)
    print("СРАВНЕНИЕ: Fine-tuning vs Prompt Design vs Prompt Tuning")
    print("Задача: SuperGLUE CB (классификация на 3 класса)")
    print(f"Модель: {config.MODEL_NAME}")
    print("=" * 80)

    # Загружаем датасет
    print("\nЗагружаем датасет...")
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    preprocessor = DataPreprocessor(tokenizer)
    dataset = preprocessor.prepare_dataset()

    print(f"Train: {len(dataset['train'])} примеров")
    print(f"Validation: {len(dataset['validation'])} примеров")
    if 'test' in dataset:
        print(f"Test: {len(dataset['test'])} примеров")
    else:
        print("⚠️ Test датасет не найден в SuperGLUE CB (используем validation)")
        test_dataset = dataset['validation']

    # Если есть test, используем его, иначе используем validation
    val_dataset = dataset['validation']
    test_dataset = dataset.get('test', dataset['validation'])

    # Создаём все три подхода
    print("\nСоздаём подходы...")
    approaches = {
        'fine_tuning': FineTuningApproach(config.MODEL_NAME, config),
        'prompt_design': PromptDesignApproach(config.MODEL_NAME, config),
        'prompt_tuning': PromptTuningApproach(config.MODEL_NAME, config),
    }

    # Сравнение параметров
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ПАРАМЕТРОВ")
    print("=" * 80)

    for name, approach in approaches.items():
        params = approach.count_parameters()
        print(f"\n{name.upper()}:")
        print(f"  Всего параметров:        {params['total']:>12,}")
        print(f"  Обучаемых параметров:    {params['trainable']:>12,}")
        print(f"  Процент обучаемых:       {params['trainable_pct']:>12.6f}%")
        if 'prompt_tokens' in params:
            print(f"  Токенов подсказки:       {params['prompt_tokens']:>12}")

    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    for approach_name, approach in approaches.items():
        if approach_name == 'prompt_design':
            print(f"\n{approach_name.upper()}: Пропускаем обучение (не требуется)")
            continue
        
        print(f"\n🚀 Обучаем {approach_name.upper()}...")
        
        trainer = approach.prepare_trainer(dataset['train'], val_dataset)
        trainer.train()
        
        print(f"✓ {approach_name.upper()} обучена!")

    # ✅ ТЕСТИРОВАНИЕ НА TEST ДАТАСЕТЕ
    print("\n" + "=" * 80)
    print("НАЧАЛО ТЕСТИРОВАНИЯ НА VAL ДАТАСЕТЕ")
    print("=" * 80)

    test_results = compare_all_approaches(approaches, val_dataset, tokenizer)

    # Сохраняем результаты
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)

if __name__ == "__main__":
    main()