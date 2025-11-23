#!/bin/bash
# Section 5 Quick Start Commands
# Все команды для запуска экспериментов и анализа

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║        Section 5: Overfitting Hypothesis - Quick Start Guide           ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

# ============================================================================
# SETUP: Установка и проверка
# ============================================================================

echo ""
echo "[SETUP] Installing dependencies..."
pip install transformers datasets evaluate torch matplotlib seaborn pandas

echo ""
echo "[SETUP] Checking Python version..."
python --version

echo ""
echo "[SETUP] Verifying imports..."
python -c "
try:
    import transformers
    import datasets
    import torch
    import matplotlib
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
"

# ============================================================================
# QUICK START: Быстрые варианты запуска
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  QUICK START OPTIONS                                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "Option 1: Run QA experiments only (10-15 minutes)"
echo "  $ cd experiments/section5/"
echo "  $ python run_section5.py --qa-only"
echo ""

echo "Option 2: Run all experiments (30-40 minutes)"
echo "  $ cd experiments/section5/"
echo "  $ python run_section5.py"
echo ""

echo "Option 3: Analyze existing results (1 minute)"
echo "  $ cd experiments/section5/"
echo "  $ python run_section5.py --no-experiments"
echo ""

echo "Option 4: Run custom experiment"
echo "  $ python -c '"
echo "from experiments_section5 import QAExperiment, ExperimentConfig"
echo "config = ExperimentConfig(...)"
echo "exp = QAExperiment(config)"
echo "results = exp.run()"
echo "'"
echo ""

# ============================================================================
# EXPLORATION: Для просмотра результатов
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  VIEWING RESULTS                                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "View QA report:"
echo "  $ cat results/qa_overfitting_report.txt"
echo ""

echo "View Paraphrase report:"
echo "  $ cat results/paraphrase_overfitting_report.txt"
echo ""

echo "View final summary:"
echo "  $ cat results/section5_summary_report.txt"
echo ""

echo "View raw metrics (JSON):"
echo "  $ python -m json.tool results/qa_model_tuning/results.json"
echo ""

echo "List all result files:"
echo "  $ find results/ -type f"
echo ""

# ============================================================================
# ANALYSIS: Для более детального анализа
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  ANALYSIS SCRIPTS                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "Python script for custom analysis:"
echo ""
echo "  from analysis_overfitting import OverfittingAnalyzer"
echo "  analyzer = OverfittingAnalyzer(results_dir='./results')"
echo ""
echo "  # Compute overfitting metrics"
echo "  gaps = analyzer.compute_train_test_gap("
echo "    'qa_model_tuning', 'squad', ['squad_dev', 'mrqa_newsqa']"
echo "  )"
echo ""
echo "  # Visualize"
echo "  analyzer.visualize_distribution_shift("
echo "    'qa_model_tuning', 'mt_distribution_shift.png'"
echo "  )"
echo ""
echo "  # Generate report"
echo "  analyzer.generate_report("
echo "    'qa_model_tuning', 'qa_prompt_tuning', 'report.txt'"
echo "  )"
echo ""

# ============================================================================
# TROUBLESHOOTING: Решение проблем
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  TROUBLESHOOTING                                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "Problem: CUDA Out of Memory"
echo "Solution:"
echo "  - Reduce batch_size: config.batch_size = 16"
echo "  - Reduce max_source_length: config.max_source_length = 256"
echo ""

echo "Problem: Low metrics"
echo "Solution:"
echo "  - Try different learning_rate (1e-4, 5e-4)"
echo "  - Try different prompt_length (10, 15, 30)"
echo "  - Increase num_epochs"
echo ""

echo "Problem: Slow training"
echo "Solution:"
echo "  - Use smaller model: 't5-small' instead of 't5-base'"
echo "  - Use GPU: CUDA_VISIBLE_DEVICES=0"
echo "  - Increase batch_size (if GPU memory allows)"
echo ""

echo "Problem: Results not reproducible"
echo "Solution:"
echo "  - Use seed=42: config.seed = 42"
echo "  - Run multiple times and average results"
echo ""

# ============================================================================
# EXPECTED OUTPUT: Ожидаемые результаты
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  EXPECTED RESULTS                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "If hypothesis is confirmed:"
echo "✓ Prompt Tuning lower train-test gap"
echo "✓ Prompt Tuning more robust to distribution shift"
echo "✓ Prompt Tuning lower spurious correlation (CV)"
echo "✓ Prompt Tuning uses 500x fewer parameters"
echo ""

echo "Expected metrics:"
echo "┌──────────────────┬──────────────────┬──────────────────┐"
echo "│ Metric           │ Model Tuning     │ Prompt Tuning    │"
echo "├──────────────────┼──────────────────┼──────────────────┤"
echo "│ SQuAD F1         │ 0.92             │ 0.88             │"
echo "│ MRQA OOD F1      │ 0.65 (-27%)      │ 0.82 (-7%)  ✓    │"
echo "│ Train-Test Gap   │ 0.27             │ 0.08        ✓    │"
echo "│ Trainable Params │ 222M             │ 50K (0.02%) ✓    │"
echo "│ Spurious (CV)    │ 0.35             │ 0.12        ✓    │"
echo "└──────────────────┴──────────────────┴──────────────────┘"
echo ""

# ============================================================================
# WORKFLOW: Типичный workflow
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  TYPICAL WORKFLOW                                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "Step 1: Quick test (5 minutes)"
echo "  $ python run_section5.py --qa-only"
echo ""

echo "Step 2: Check results"
echo "  $ cat results/qa_overfitting_report.txt"
echo ""

echo "Step 3: Full run (30-40 minutes)"
echo "  $ python run_section5.py"
echo ""

echo "Step 4: Generate summary"
echo "  $ cat results/section5_summary_report.txt"
echo ""

echo "Step 5: Use in publication"
echo "  - Copy graphs: results/*.png"
echo "  - Copy table: results/section5_summary_report.txt"
echo ""

# ============================================================================
# DOCUMENTATION: Документация
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  DOCUMENTATION FILES                                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📖 SECTION5_README.md"
echo "   Complete documentation, usage guide, and FAQ"
echo ""

echo "📖 EXAMPLES_AND_QUICK_START.py"
echo "   10 practical examples with code"
echo ""

echo "📖 INTEGRATION_GUIDE.md"
echo "   How to integrate into your existing repository"
echo ""

echo "📖 PROJECT_STRUCTURE_GUIDE.txt"
echo "   File-by-file explanation of all components"
echo ""

echo "📖 This file (quick start commands)"
echo "   All bash commands for running experiments"
echo ""

# ============================================================================
# NEXT STEPS
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  NEXT STEPS                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "1. Read SECTION5_README.md for detailed documentation"
echo ""

echo "2. Look at EXAMPLES_AND_QUICK_START.py for code examples"
echo ""

echo "3. Run quick test:"
echo "   cd experiments/section5/"
echo "   python run_section5.py --qa-only"
echo ""

echo "4. Check results:"
echo "   cat results/qa_overfitting_report.txt"
echo ""

echo "5. Integrate into your repository (see INTEGRATION_GUIDE.md)"
echo ""

echo "6. Run full experiments and collect results"
echo ""

echo "7. Publish results or use in reports/papers"
echo ""

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  Ready to start? Run:                                                  ║"
echo "║                                                                        ║"
echo "║  cd experiments/section5/                                              ║"
echo "║  python run_section5.py --qa-only                                      ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
