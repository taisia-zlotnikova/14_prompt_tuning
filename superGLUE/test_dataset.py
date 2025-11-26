# Dataset Debugging & Testing Script
# Используйте этот скрипт для локального тестирования подготовки датасета

import torch
from transformers import T5Tokenizer
from datasets import load_dataset
import pandas as pd


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

MODEL_NAME = "t5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 10


# ============================================================================
# ШАБЛОН LABEL MAP
# ============================================================================

LABEL_MAP = {
    0: "entailment",
    1: "contradiction",
    2: "neutral"
}


# ============================================================================
# ФУНКЦИЯ 1: ЗАГРУЗКА И ПРОСМОТР ИСХОДНЫХ ДАННЫХ
# ============================================================================

def load_and_inspect_raw_dataset():
    """Загружаем датасет и смотрим его структуру"""
    print("\n" + "="*80)
    print("ШАГ 1: ЗАГРУЗКА ИСХОДНОГО ДАТАСЕТА")
    print("="*80)
    
    dataset = load_dataset("super_glue", "cb")
    
    print(f"\n✓ Датасет загружен!")
    print(f"\nДоступные сплиты: {list(dataset.keys())}")
    
    # Размеры
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} примеров")
    
    # Структура
    print(f"\n📋 Колонки в датасете: {dataset['train'].column_names}")
    print(f"\n📊 Типы данных:")
    print(dataset['train'].features)
    
    # Примеры
    print(f"\n📌 Примеры из train сплита:")
    print("-" * 80)
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        print(f"\nПример {i}:")
        print(f"  premise:    {example['premise'][:80]}")
        print(f"  hypothesis: {example['hypothesis'][:80]}")
        print(f"  label:      {example['label']} ({LABEL_MAP.get(example['label'], 'UNKNOWN')})")
        print(f"  idx:        {example['idx']}")
    
    # Статистика labels
    print(f"\n📊 Распределение labels:")
    label_counts = {}
    for example in dataset['train']:
        label = example['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        name = LABEL_MAP.get(label, "UNKNOWN")
        percentage = 100 * count / len(dataset['train'])
        print(f"  label {label} ({name}): {count} примеров ({percentage:.1f}%)")
    
    return dataset


# ============================================================================
# ФУНКЦИЯ 2: ФИЛЬТРАЦИЯ INVALID LABELS
# ============================================================================

def filter_invalid_labels(dataset):
    """Фильтруем примеры с label = -1"""
    print("\n" + "="*80)
    print("ШАГ 2: ФИЛЬТРАЦИЯ INVALID LABELS")
    print("="*80)
    
    print(f"\nДо фильтрации:")
    print(f"  Train:      {len(dataset['train'])} примеров")
    print(f"  Validation: {len(dataset['validation'])} примеров")
    
    # Проверяем есть ли invalid labels
    invalid_in_train = sum(1 for ex in dataset['train'] if ex['label'] not in LABEL_MAP)
    invalid_in_val = sum(1 for ex in dataset['validation'] if ex['label'] not in LABEL_MAP)
    
    print(f"\nПримеры с invalid labels:")
    print(f"  Train:      {invalid_in_train} примеров")
    print(f"  Validation: {invalid_in_val} примеров")
    
    if invalid_in_train > 0:
        print(f"\n⚠️ Примеры с invalid labels в train:")
        for i, ex in enumerate(dataset['train']):
            if ex['label'] not in LABEL_MAP:
                print(f"  Пример {i}: label={ex['label']}")
                if i >= 4:  # Показываем максимум 5
                    print("  ...")
                    break
    
    # Фильтруем
    print(f"\n🔍 Фильтруем...")
    dataset = dataset.filter(lambda x: x['label'] in LABEL_MAP, desc="Filtering invalid labels")
    
    print(f"\nПосле фильтрации:")
    print(f"  Train:      {len(dataset['train'])} примеров")
    print(f"  Validation: {len(dataset['validation'])} примеров")
    
    return dataset


# ============================================================================
# ФУНКЦИЯ 3: ФОРМАТИРОВАНИЕ ДЛЯ T5
# ============================================================================

def format_for_t5(example):
    """Преобразуем в T5 text-to-text формат"""
    input_text = f"premise: {example['premise']} hypothesis: {example['hypothesis']}"
    target_text = LABEL_MAP[example['label']]
    
    return {
        'input_text': input_text,
        'target_text': target_text,
        'label': example['label'],  # Сохраняем для отладки
    }


def apply_formatting(dataset):
    """Применяем форматирование"""
    print("\n" + "="*80)
    print("ШАГ 3: ФОРМАТИРОВАНИЕ ДЛЯ T5")
    print("="*80)
    
    print(f"\nКолонки ДО форматирования: {dataset['train'].column_names}")
    
    # Применяем форматирование
    formatted_dataset = dataset.map(format_for_t5, desc="Formatting for T5")
    
    print(f"Колонки ПОСЛЕ форматирования: {formatted_dataset['train'].column_names}")
    
    # Показываем примеры
    print(f"\n📌 Примеры после форматирования:")
    print("-" * 80)
    for i in range(min(2, len(formatted_dataset['train']))):
        example = formatted_dataset['train'][i]
        print(f"\nПример {i}:")
        print(f"  input_text:  {example['input_text'][:100]}...")
        print(f"  target_text: {example['target_text']}")
        print(f"  label:       {example['label']}")
    
    return formatted_dataset


# ============================================================================
# ФУНКЦИЯ 4: ТОКЕНИЗАЦИЯ
# ============================================================================

def tokenize_dataset(dataset):
    """Токенизируем датасет"""
    print("\n" + "="*80)
    print("ШАГ 4: ТОКЕНИЗАЦИЯ")
    print("="*80)
    
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    print(f"\n✓ Tokenizer загружен: {MODEL_NAME}")
    print(f"  Размер словаря: {tokenizer.vocab_size}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    
    print(f"\nКолонки ДО токенизации: {dataset['train'].column_names}")
    
    # Функция токенизации
    def preprocess_function(examples):
        inputs = examples['input_text']
        targets = examples['target_text']
        
        # Токенизируем входы
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None,  # НЕ возвращаем тензоры (потом в Trainer)
        )
        
        # Токенизируем целевые значения
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        
        # Для T5: labels = input_ids целей
        model_inputs["labels"] = labels["input_ids"]
        
        # Заменяем pad tokens (-100 игнорируется в loss)
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in model_inputs["labels"]
        ]
        
        return model_inputs
    
    # Применяем токенизацию
    print(f"\n🔍 Применяем токенизацию...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing"
    )
    
    print(f"Колонки ПОСЛЕ токенизации: {tokenized_dataset['train'].column_names}")
    
    # Показываем примеры
    print(f"\n📌 Примеры после токенизации:")
    print("-" * 80)
    for i in range(min(2, len(tokenized_dataset['train']))):
        example = tokenized_dataset['train'][i]
        print(f"\nПример {i}:")
        print(f"  input_ids shape:   {len(example['input_ids'])}")
        print(f"  attention_mask:    {example['attention_mask'][:20]}... (первые 20)")
        print(f"  labels shape:      {len(example['labels'])}")
        print(f"  labels:            {example['labels'][:20]}... (первые 20)")
    
    return tokenized_dataset, tokenizer


# ============================================================================
# ФУНКЦИЯ 5: ВАЛИДАЦИЯ ДАТАСЕТА
# ============================================================================

def validate_dataset(dataset, tokenizer):
    """Проверяем корректность подготовленного датасета"""
    print("\n" + "="*80)
    print("ШАГ 5: ВАЛИДАЦИЯ ДАТАСЕТА")
    print("="*80)
    
    # Проверяем обязательные колонки
    required_columns = ['input_ids', 'attention_mask', 'labels']
    print(f"\n✓ Проверяем обязательные колонки...")
    for col in required_columns:
        if col in dataset['train'].column_names:
            print(f"  ✓ {col} присутствует")
        else:
            print(f"  ✗ {col} отсутствует!")
    
    # Проверяем размеры
    print(f"\n✓ Проверяем размеры примеров...")
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        input_ids_len = len(example['input_ids'])
        labels_len = len(example['labels'])
        
        if input_ids_len != MAX_INPUT_LENGTH:
            print(f"  ⚠️ Пример {i}: input_ids length={input_ids_len} (ожидалось {MAX_INPUT_LENGTH})")
        if labels_len != MAX_TARGET_LENGTH:
            print(f"  ⚠️ Пример {i}: labels length={labels_len} (ожидалось {MAX_TARGET_LENGTH})")
    
    print(f"  ✓ Размеры корректны")
    
    # Проверяем значения
    print(f"\n✓ Проверяем значения токенов...")
    example = dataset['train'][0]
    
    input_ids = example['input_ids']
    print(f"  Input IDs диапазон: min={min(input_ids)}, max={max(input_ids)}")
    print(f"  (должны быть в диапазоне [0, {tokenizer.vocab_size-1}])")
    
    labels = [l for l in example['labels'] if l != -100]
    if labels:
        print(f"  Labels диапазон: min={min(labels)}, max={max(labels)}")
    else:
        print(f"  ⚠️ Все labels = -100 (это нормально для некоторых примеров)")
    
    # Проверяем attention_mask
    print(f"\n✓ Проверяем attention_mask...")
    attn = example['attention_mask']
    unique_values = set(attn)
    print(f"  Уникальные значения: {unique_values}")
    if unique_values == {0, 1}:
        print(f"  ✓ Корректно")
    else:
        print(f"  ✗ Ошибка! Должны быть только 0 и 1")
    
    # Общая статистика
    print(f"\n📊 Общая статистика:")
    print(f"  Train примеров:      {len(dataset['train'])}")
    print(f"  Validation примеров: {len(dataset['validation'])}")
    print(f"  Колонки:             {dataset['train'].column_names}")
    
    print(f"\n✓ Датасет валидн!")


# ============================================================================
# ФУНКЦИЯ 6: ВИЗУАЛИЗАЦИЯ
# ============================================================================

def visualize_dataset(dataset, tokenizer):
    """Визуализируем несколько примеров"""
    print("\n" + "="*80)
    print("ШАГ 6: ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ")
    print("="*80)
    
    print(f"\nПримеры из датасета с декодированием:")
    print("-" * 80)
    
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        
        # Декодируем
        input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
        labels = [l for l in example['labels'] if l != -100]
        target_text = tokenizer.decode(labels, skip_special_tokens=False) if labels else "[ALL PADDING]"
        
        print(f"\nПример {i}:")
        print(f"  Input:  {input_text[:100]}...")
        print(f"  Target: {target_text}")
        print(f"  Input shape:  {len(example['input_ids'])}")
        print(f"  Target shape: {len(example['labels'])}")


# ============================================================================
# MAIN: ЗАПУСТИТЬ ВСЕ ТЕСТЫ
# ============================================================================

def main():
    print("\n" + "🧪" * 40)
    print("ТЕСТИРОВАНИЕ И ОТЛАДКА ПОДГОТОВКИ ДАТАСЕТА SuperGLUE CB")
    print("🧪" * 40)
    
    try:
        # Шаг 1: Загрузка
        dataset = load_and_inspect_raw_dataset()
        
        # Шаг 2: Фильтрация
        dataset = filter_invalid_labels(dataset)
        
        # Шаг 3: Форматирование
        dataset = apply_formatting(dataset)
        
        # Шаг 4: Токенизация
        dataset, tokenizer = tokenize_dataset(dataset)
        
        # Шаг 5: Валидация
        validate_dataset(dataset, tokenizer)
        
        # Шаг 6: Визуализация
        visualize_dataset(dataset, tokenizer)
        
        print("\n" + "✅" * 40)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("✅" * 40)
        
        return dataset, tokenizer
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    dataset, tokenizer = main()
