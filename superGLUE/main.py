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
    MODEL_NAME = "google/t5-small"  # или "google/t5-base"
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
        
        return model_inputs
    
    def prepare_dataset(self):
        """Загружаем и подготавливаем датасет"""
        dataset = self.load_superglue_cb()
        
        # Применяем форматирование
        dataset = dataset.map(
            self.format_for_t5,
            remove_columns=dataset['train'].column_names
        )
        
        # Токенизируем
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['input_text', 'target_text'],
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
    
    # Создаём все три подхода
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
    
    # Ожидаемые результаты
    print("\n" + "=" * 80)
    print("ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ на SuperGLUE CB")
    print("=" * 80)
    print("""
Fine-tuning:
  ✓ Точность: ~85-90% (лучший результат)
  ✓ F1: ~83-88%
  ✗ Время обучения: 30-60 минут
  ✗ Память: 11GB+ для T5-XXL
  
Prompt Design:
  ✓ Время: 0 (нет обучения)
  ✓ Память: минимум
  ✗ Точность: ~50-60% (плохо, требует хорошего prompt engineering)
  ✗ F1: ~45-55%
  
Prompt Tuning:
  ✓ Точность: ~80-87% (близко к fine-tuning!)
  ✓ F1: ~78-85%
  ✓ Время обучения: 5-10 минут (в 5-10 раз быстрее!)
  ✓ Память: ~100MB (в 100 раз меньше!)
  ✓ Одна модель для всех задач
""")

if __name__ == "__main__":
    main()
