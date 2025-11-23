"""
Experiment: Model Tuning vs Prompt Tuning - Overfitting Hypothesis
Section 5: Comparing generalization on distribution shift tasks

Задача: Проверить гипотезу о переобучении при полном fine-tuning
- Model Tuning (все параметры обновляются) -> высокий риск spurious correlations
- Prompt Tuning (заморожены параметры LM) -> более устойчивы к переобучению
"""

import os
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import evaluate

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    experiment_name: str
    task_type: str  # "qa" или "paraphrase"
    model_name: str
    tuning_approach: str  # "model_tuning" или "prompt_tuning"
    
    # Dataset config
    train_dataset: str
    eval_datasets: List[str]
    
    # Model config
    max_source_length: int = 512
    max_target_length: int = 50
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 1e-3
    warmup_steps: int = 500
    
    # Prompt tuning specific
    prompt_length: int = 20
    init_text: str = ""
    
    # Output
    output_dir: str = "./results"
    seed: int = 42
    

class PromptTuningLayer(torch.nn.Module):
    """Слой для prompt tuning: обучаемые префиксы"""
    
    def __init__(self, prompt_length: int, hidden_size: int, vocab_size: int = None):
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        
        # Инициализация обучаемых префиксов (soft prompts)
        self.soft_prompt = torch.nn.Parameter(
            torch.randn(1, prompt_length, hidden_size) * 0.02
        )
        
    def forward(self, encoder_outputs, attention_mask=None):
        """Конкатенирует soft prompt с encoder output"""
        batch_size = encoder_outputs.shape[0]
        
        # Расширяем prompt для всего batch
        soft_prompt = self.soft_prompt.expand(batch_size, -1, -1)
        
        # Конкатенируем: [soft_prompt; encoder_hidden_states]
        extended_outputs = torch.cat([soft_prompt, encoder_outputs], dim=1)
        
        # Расширяем attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.prompt_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            return extended_outputs, extended_mask
        
        return extended_outputs, attention_mask


class ModelTuningWrapper(torch.nn.Module):
    """Обертка для model tuning (все параметры обновляются)"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Все параметры trainable - никакого замораживания
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class PromptTuningWrapper(torch.nn.Module):
    """Обертка для prompt tuning (замораживаются параметры LM)"""
    
    def __init__(self, model, prompt_length: int):
        super().__init__()
        self.model = model
        self.prompt_tuning = PromptTuningLayer(
            prompt_length=prompt_length,
            hidden_size=model.config.d_model
        )
        
        # Замораживаем основные параметры модели
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Разморозим только prompt-tuning параметры
        for param in self.prompt_tuning.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids=None, attention_mask=None, 
                decoder_input_ids=None, labels=None, **kwargs):
        # Получаем encoder outputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Добавляем soft prompts
        last_hidden_state, extended_mask = self.prompt_tuning(
            encoder_outputs.last_hidden_state,
            attention_mask
        )
        
        # Передаем в decoder
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=kwargs.get('decoder_attention_mask'),
            encoder_hidden_states=last_hidden_state,
            encoder_attention_mask=extended_mask,
            return_dict=True
        )
        
        # Получаем логиты и lm_loss
        lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': lm_logits,
        }
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class ExperimentTracker:
    """Трекер метрик для анализа переобучения"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = defaultdict(lambda: defaultdict(list))
        
    def record(self, dataset_name: str, metric_name: str, value: float):
        """Записывает метрику"""
        self.metrics[dataset_name][metric_name].append(value)
    
    def compute_overfitting_metrics(self, 
                                   train_dataset: str,
                                   eval_datasets: List[str]) -> Dict[str, float]:
        """
        Анализирует переобучение:
        - train_test_gap: разница между train и test метриками
        - distribution_shift: деградация при OOD датасетах
        """
        results = {}
        
        # Средние метрики на тренировочном датасете
        train_scores = {
            metric: np.mean(values)
            for metric, values in self.metrics[train_dataset].items()
        }
        
        # Для каждого eval датасета
        for eval_ds in eval_datasets:
            eval_scores = {
                metric: np.mean(values)
                for metric, values in self.metrics[eval_ds].items()
            }
            
            # Gap = |train - eval|
            for metric in train_scores:
                if metric in eval_scores:
                    gap = train_scores[metric] - eval_scores[metric]
                    results[f"{eval_ds}_{metric}_gap"] = gap
        
        return results
    
    def to_dict(self) -> Dict:
        """Экспортирует метрики"""
        return {
            ds: {metric: values 
                 for metric, values in metrics.items()}
            for ds, metrics in self.metrics.items()
        }


class QAExperiment:
    """Эксперимент 1: Question Answering на MRQA"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Выбираем подход
        if config.tuning_approach == "prompt_tuning":
            self.model = PromptTuningWrapper(self.base_model, config.prompt_length)
        else:  # model_tuning
            self.model = ModelTuningWrapper(self.base_model)
        
        self.tracker = ExperimentTracker(config.experiment_name)
        
        # Логируем количество trainable параметров
        trainable = self.model.get_trainable_params()
        total = sum(p.numel() for p in self.base_model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def prepare_dataset(self, dataset_name: str, split: str = "train"):
        """Загружает и препроцессирует датасет"""
        
        # Маппирование имен датасетов
        dataset_mapping = {
            "squad": "squad",
            "squad_dev": ("squad", "validation"),
            "mrqa_newsqa": "mrqa",  # с фильтром по источнику
            "mrqa_searchqa": "mrqa",
            "mrqa_triviaqa": "mrqa",
            "mrqa_hotpotqa": "mrqa",
        }
        
        if dataset_name == "squad":
            dataset = load_dataset("squad", split=split)
        elif "mrqa" in dataset_name:
            # Для MRQA используем фильтр по источнику
            source = dataset_name.replace("mrqa_", "")
            dataset = load_dataset("mrqa", split=split)
            dataset = dataset.filter(lambda x: x["source"] == source)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Препроцессинг: question + context -> answer
        def preprocess(examples):
            inputs = [
                f"question: {q} context: {c}"
                for q, c in zip(examples["question"], examples["context"])
            ]
            targets = [a["text"][0] for a in examples["answers"]]
            
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config.max_source_length,
                truncation=True,
                padding="max_length"
            )
            
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_target_length,
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = dataset.map(preprocess, batched=True)
        dataset.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "labels"
        ])
        
        return dataset
    
    def run(self):
        """Запускает эксперимент"""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        # Загружаем тренировочный датасет
        train_dataset = self.prepare_dataset(self.config.train_dataset, split="train")
        
        # Определяем output directory
        exp_dir = Path(self.config.output_dir) / self.config.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Конфигурация обучения
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(exp_dir / "checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            logging_steps=50,
            report_to=["tensorboard"],
            seed=self.config.seed,
            predict_with_generate=True,
            logging_dir=str(exp_dir / "logs"),
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model.model if isinstance(self.model, ModelTuningWrapper) else self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
            tokenizer=self.tokenizer,
        )
        
        # Обучение
        trainer.train()
        
        # Оценка на разных датасетах
        metric = evaluate.load("squad")
        
        for eval_dataset_name in self.config.eval_datasets:
            logger.info(f"Evaluating on {eval_dataset_name}...")
            eval_dataset = self.prepare_dataset(eval_dataset_name, split="validation")
            
            predictions = trainer.predict(eval_dataset)
            decoded_preds = self.tokenizer.batch_decode(
                predictions.predictions, skip_special_tokens=True
            )
            
            # Загружаем references
            ref_dataset = self.prepare_dataset(eval_dataset_name, split="validation")
            references = [
                [self.tokenizer.decode(label_id, skip_special_tokens=True) 
                 for label_id in ref]
                for ref in ref_dataset["labels"]
            ]
            
            # Вычисляем метрики
            results = metric.compute(predictions=decoded_preds, references=references)
            
            self.tracker.record(eval_dataset_name, "exact_match", results["exact_match"])
            self.tracker.record(eval_dataset_name, "f1", results["f1"])
            
            logger.info(f"{eval_dataset_name}: EM={results['exact_match']:.2f}, F1={results['f1']:.2f}")
        
        # Анализ переобучения
        overfitting_metrics = self.tracker.compute_overfitting_metrics(
            self.config.train_dataset,
            self.config.eval_datasets
        )
        
        # Сохраняем результаты
        results_file = exp_dir / "results.json"
        results = {
            "config": asdict(self.config),
            "metrics": self.tracker.to_dict(),
            "overfitting_analysis": overfitting_metrics,
            "trainable_params": self.model.get_trainable_params(),
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        return results


class ParaphraseExperiment:
    """Эксперимент 2: Paraphrase Detection (QQP + MRPC)"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Выбираем подход
        if config.tuning_approach == "prompt_tuning":
            self.model = PromptTuningWrapper(self.base_model, config.prompt_length)
        else:  # model_tuning
            self.model = ModelTuningWrapper(self.base_model)
        
        self.tracker = ExperimentTracker(config.experiment_name)
        
        # Логируем
        trainable = self.model.get_trainable_params()
        total = sum(p.numel() for p in self.base_model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def prepare_dataset(self, dataset_name: str, split: str = "train"):
        """Загружает и препроцессирует датасет"""
        
        if dataset_name == "qqp":
            dataset = load_dataset("glue", "qqp", split=split)
        elif dataset_name == "mrpc":
            dataset = load_dataset("glue", "mrpc", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Препроцессинг: sentence1 + sentence2 -> label
        def preprocess(examples):
            inputs = [
                f"sentence1: {s1} sentence2: {s2}"
                for s1, s2 in zip(examples["sentence1"], examples["sentence2"])
            ]
            targets = ["equivalent" if label == 1 else "not_equivalent" 
                      for label in examples["label"]]
            
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config.max_source_length,
                truncation=True,
                padding="max_length"
            )
            
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_target_length,
                truncation=True,
                padding="max_length"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = dataset.map(preprocess, batched=True)
        dataset.set_format(type="torch", columns=[
            "input_ids", "attention_mask", "labels"
        ])
        
        return dataset
    
    def run_cross_dataset_eval(self):
        """Запускает кросс-датасетное оценивание"""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        results_all = {}
        
        # Для каждой пары train-eval
        train_eval_pairs = [
            ("qqp", ["qqp", "mrpc"]),
            ("mrpc", ["mrpc", "qqp"]),
        ]
        
        for train_ds, eval_dss in train_eval_pairs:
            exp_name = f"{self.config.experiment_name}_{train_ds}"
            exp_dir = Path(self.config.output_dir) / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Загружаем тренировочный датасет
            train_dataset = self.prepare_dataset(train_ds, split="train")
            
            # Конфигурация обучения
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(exp_dir / "checkpoints"),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                save_total_limit=2,
                logging_steps=50,
                report_to=[],  # без tensorboard для простоты
                seed=self.config.seed,
            )
            
            trainer = Seq2SeqTrainer(
                model=self.model.model if isinstance(self.model, ModelTuningWrapper) else self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=DataCollatorForSeq2Seq(self.tokenizer),
            )
            
            # Обучение
            trainer.train()
            
            # Оценка
            metric = evaluate.load("glue", "mrpc")  # используем MRPC метрики (accuracy, f1)
            
            for eval_ds in eval_dss:
                logger.info(f"Evaluating {train_ds} model on {eval_ds}...")
                eval_dataset = self.prepare_dataset(eval_ds, split="validation")
                
                predictions = trainer.predict(eval_dataset)
                
                # Для классификации берем argmax
                pred_labels = np.argmax(predictions.predictions, axis=-1)
                ref_labels = eval_dataset["labels"]
                
                # Вычисляем метрики
                results = metric.compute(predictions=pred_labels, references=ref_labels)
                
                self.tracker.record(f"{train_ds}_train_{eval_ds}_eval", 
                                  "accuracy", results["accuracy"])
                self.tracker.record(f"{train_ds}_train_{eval_ds}_eval",
                                  "f1", results["f1"])
                
                logger.info(f"Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
            
            results_all[train_ds] = self.tracker.to_dict()
        
        # Анализ переобучения и distribution shift
        overfitting_metrics = self._analyze_distribution_shift()
        
        # Сохраняем результаты
        results_file = Path(self.config.output_dir) / f"{self.config.experiment_name}_results.json"
        results = {
            "config": asdict(self.config),
            "all_metrics": results_all,
            "distribution_shift_analysis": overfitting_metrics,
            "trainable_params": self.model.get_trainable_params(),
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        return results
    
    def _analyze_distribution_shift(self) -> Dict[str, float]:
        """Анализирует влияние distribution shift"""
        metrics_data = self.tracker.to_dict()
        analysis = {}
        
        # QQP model на MRPC: сколько падает accuracy
        qqp_mrpc_data = metrics_data.get("qqp_train_mrpc_eval", {})
        qqp_qqp_data = metrics_data.get("qqp_train_qqp_eval", {})
        
        if qqp_qqp_data and qqp_mrpc_data:
            qqp_acc = np.mean(qqp_qqp_data.get("accuracy", [0]))
            mrpc_acc = np.mean(qqp_mrpc_data.get("accuracy", [0]))
            analysis["qqp_to_mrpc_accuracy_drop"] = qqp_acc - mrpc_acc
        
        # MRPC model на QQP
        mrpc_qqp_data = metrics_data.get("mrpc_train_qqp_eval", {})
        mrpc_mrpc_data = metrics_data.get("mrpc_train_mrpc_eval", {})
        
        if mrpc_mrpc_data and mrpc_qqp_data:
            mrpc_acc = np.mean(mrpc_mrpc_data.get("accuracy", [0]))
            qqp_acc = np.mean(mrpc_qqp_data.get("accuracy", [0]))
            analysis["mrpc_to_qqp_accuracy_drop"] = mrpc_acc - qqp_acc
        
        return analysis


if __name__ == "__main__":
    # Пример запуска для Эксперимента 1: QA
    qa_config = ExperimentConfig(
        experiment_name="qa_model_tuning_vs_prompt_tuning",
        task_type="qa",
        model_name="t5-base",
        tuning_approach="model_tuning",  # попробуем оба
        train_dataset="squad",
        eval_datasets=["squad_dev", "mrqa_newsqa", "mrqa_triviaqa"],
        num_epochs=3,
        learning_rate=1e-3,
    )
    
    qa_exp = QAExperiment(qa_config)
    # qa_exp.run()
    
    # Пример запуска для Эксперимента 2: Paraphrase
    para_config = ExperimentConfig(
        experiment_name="paraphrase_model_tuning_vs_prompt_tuning",
        task_type="paraphrase",
        model_name="t5-base",
        tuning_approach="prompt_tuning",
        train_dataset="qqp",
        eval_datasets=["qqp", "mrpc"],
        num_epochs=10,
        learning_rate=1e-3,
        prompt_length=15,
        init_text="paraphrase: ",
    )
    
    para_exp = ParaphraseExperiment(para_config)
    # para_exp.run_cross_dataset_eval()
