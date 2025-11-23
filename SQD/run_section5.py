"""
Main runner для Раздела 5: Experiments on Overfitting
Запускает оба эксперимента (QA + Paraphrase) с model tuning и prompt tuning
"""

import argparse
import sys
from pathlib import Path
from experiments_section5 import (
    QAExperiment, ParaphraseExperiment, ExperimentConfig
)
from analysis_overfitting import OverfittingAnalyzer


def run_qa_experiments():
    """Запускает оба подхода для QA задачи"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Question Answering (QA) - SQuAD + MRQA 2019")
    print("="*80)
    
    # Model Tuning
    print("\n[1/4] Running Model Tuning (all parameters updated)...")
    mt_config = ExperimentConfig(
        experiment_name="qa_model_tuning",
        task_type="qa",
        model_name="t5-base",
        tuning_approach="model_tuning",
        train_dataset="squad",
        eval_datasets=["squad_dev", "mrqa_newsqa", "mrqa_triviaqa"],
        num_epochs=3,
        learning_rate=1e-3,
        batch_size=32,
    )
    
    mt_exp = QAExperiment(mt_config)
    mt_results = mt_exp.run()
    print("✓ Model Tuning completed")
    
    # Prompt Tuning
    print("\n[2/4] Running Prompt Tuning (frozen LM + learnable prompts)...")
    pt_config = ExperimentConfig(
        experiment_name="qa_prompt_tuning",
        task_type="qa",
        model_name="t5-base",
        tuning_approach="prompt_tuning",
        train_dataset="squad",
        eval_datasets=["squad_dev", "mrqa_newsqa", "mrqa_triviaqa"],
        num_epochs=3,
        learning_rate=1e-3,
        batch_size=32,
        prompt_length=20,
        init_text="question answering: ",
    )
    
    pt_exp = QAExperiment(pt_config)
    pt_results = pt_exp.run()
    print("✓ Prompt Tuning completed")
    
    return "qa_model_tuning", "qa_prompt_tuning"


def run_paraphrase_experiments():
    """Запускает оба подхода для Paraphrase задачи"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Paraphrase Detection (QQP + MRPC)")
    print("="*80)
    
    # Model Tuning
    print("\n[3/4] Running Paraphrase - Model Tuning (all parameters updated)...")
    mt_para_config = ExperimentConfig(
        experiment_name="paraphrase_model_tuning",
        task_type="paraphrase",
        model_name="t5-base",
        tuning_approach="model_tuning",
        train_dataset="qqp",
        eval_datasets=["qqp", "mrpc"],
        num_epochs=10,
        learning_rate=1e-3,
        batch_size=32,
    )
    
    mt_para_exp = ParaphraseExperiment(mt_para_config)
    mt_para_results = mt_para_exp.run_cross_dataset_eval()
    print("✓ Paraphrase Model Tuning completed")
    
    # Prompt Tuning
    print("\n[4/4] Running Paraphrase - Prompt Tuning (frozen LM + learnable prompts)...")
    pt_para_config = ExperimentConfig(
        experiment_name="paraphrase_prompt_tuning",
        task_type="paraphrase",
        model_name="t5-base",
        tuning_approach="prompt_tuning",
        train_dataset="qqp",
        eval_datasets=["qqp", "mrpc"],
        num_epochs=10,
        learning_rate=1e-3,
        batch_size=32,
        prompt_length=15,
        init_text="paraphrase: ",
    )
    
    pt_para_exp = ParaphraseExperiment(pt_para_config)
    pt_para_results = pt_para_exp.run_cross_dataset_eval()
    print("✓ Paraphrase Prompt Tuning completed")
    
    return "paraphrase_model_tuning", "paraphrase_prompt_tuning"


def analyze_results(qa_mt_exp, qa_pt_exp, para_mt_exp, para_pt_exp):
    """Анализирует результаты обоих экспериментов"""
    print("\n" + "="*80)
    print("ANALYSIS: Overfitting Hypothesis")
    print("="*80)
    
    analyzer = OverfittingAnalyzer(results_dir="./results")
    
    # Анализ Эксперимента 1: QA
    print("\n[1/2] Analyzing Experiment 1: QA...")
    print("-" * 60)
    
    qa_comparison = analyzer.compare_tuning_approaches(
        model_tuning_exp=qa_mt_exp,
        prompt_tuning_exp=qa_pt_exp
    )
    
    analyzer.visualize_distribution_shift(
        experiment_name=qa_mt_exp,
        output_file="./results/qa_mt_distribution_shift.png"
    )
    analyzer.visualize_distribution_shift(
        experiment_name=qa_pt_exp,
        output_file="./results/qa_pt_distribution_shift.png"
    )
    
    analyzer.generate_report(
        model_tuning_exp=qa_mt_exp,
        prompt_tuning_exp=qa_pt_exp,
        output_file="./results/qa_overfitting_report.txt"
    )
    
    # Анализ Эксперимента 2: Paraphrase
    print("\n[2/2] Analyzing Experiment 2: Paraphrase...")
    print("-" * 60)
    
    para_comparison = analyzer.compare_tuning_approaches(
        model_tuning_exp=para_mt_exp,
        prompt_tuning_exp=para_pt_exp
    )
    
    analyzer.generate_report(
        model_tuning_exp=para_mt_exp,
        prompt_tuning_exp=para_pt_exp,
        output_file="./results/paraphrase_overfitting_report.txt"
    )
    
    # Общий отчет
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    with open("./results/section5_summary_report.txt", "w") as f:
        f.write("SECTION 5: OVERFITTING HYPOTHESIS - SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("HYPOTHESIS:\n")
        f.write("When all model parameters are updated during fine-tuning (model tuning),\n")
        f.write("the model can memorize dataset-specific lexical signals and spurious\n")
        f.write("correlations. In contrast, prompt tuning freezes the LM parameters,\n")
        f.write("allowing only prompt parameters to be updated, leading to better\n")
        f.write("generalization on out-of-distribution tasks.\n\n")
        
        f.write("EXPERIMENT 1: Question Answering (SQuAD + MRQA 2019)\n")
        f.write("-"*80 + "\n")
        f.write("Comparison Metrics:\n")
        f.write(str(qa_comparison) + "\n\n")
        
        f.write("EXPERIMENT 2: Paraphrase Detection (QQP + MRPC)\n")
        f.write("-"*80 + "\n")
        f.write("Comparison Metrics:\n")
        f.write(str(para_comparison) + "\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("1. Train-test gap analysis: Shows if model memorized training data\n")
        f.write("2. Distribution shift impact: Shows robustness to OOD data\n")
        f.write("3. Spurious correlation proxy: Coefficient of variation across datasets\n")
        f.write("   - Higher CV = model learned dataset-specific features\n")
        f.write("   - Lower CV = model learned more general features\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write("- Review QA and Paraphrase reports in ./results/\n")
        f.write("- Compare trainable parameter counts\n")
        f.write("- Analyze attention patterns for spurious correlations\n")
    
    print("\n✓ Analysis complete. Reports saved to ./results/")


def main():
    parser = argparse.ArgumentParser(
        description="Section 5: Model Tuning vs Prompt Tuning - Overfitting Hypothesis"
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Run only QA experiment"
    )
    parser.add_argument(
        "--paraphrase-only",
        action="store_true",
        help="Run only Paraphrase experiment"
    )
    parser.add_argument(
        "--no-experiments",
        action="store_true",
        help="Skip experiments, run analysis only (requires pre-existing results)"
    )
    
    args = parser.parse_args()
    
    # Создаем directory для результатов
    Path("./results").mkdir(exist_ok=True)
    
    qa_mt_exp = None
    qa_pt_exp = None
    para_mt_exp = None
    para_pt_exp = None
    
    if not args.no_experiments:
        if not args.paraphrase_only:
            qa_mt_exp, qa_pt_exp = run_qa_experiments()
        
        if not args.qa_only:
            para_mt_exp, para_pt_exp = run_paraphrase_experiments()
    else:
        # Используем предыдущие результаты
        qa_mt_exp = "qa_model_tuning"
        qa_pt_exp = "qa_prompt_tuning"
        para_mt_exp = "paraphrase_model_tuning"
        para_pt_exp = "paraphrase_prompt_tuning"
    
    # Анализ
    if qa_mt_exp or para_mt_exp:
        analyze_results(
            qa_mt_exp or "qa_model_tuning",
            qa_pt_exp or "qa_prompt_tuning",
            para_mt_exp or "paraphrase_model_tuning",
            para_pt_exp or "paraphrase_prompt_tuning"
        )
    
    print("\n" + "="*80)
    print("✓ All experiments completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
