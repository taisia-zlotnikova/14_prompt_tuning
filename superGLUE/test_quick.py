# Quick Sanity Check Script - Testing Code Workability
# Быстрая проверка на работоспособность в 1-2 минуты

import torch
import sys

print("="*80)
print("БЫСТРАЯ ПРОВЕРКА КОДА НА РАБОТОСПОСОБНОСТЬ")
print("="*80)

# 1. Check PyTorch
print("\n[1/6] Checking PyTorch...")
try:
    print(f"      PyTorch version: {torch.__version__}")
    print(f"      GPU available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Using device: {device}")
    print("      ✓ PyTorch OK")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# 2. Check Transformers
print("\n[2/6] Loading T5 model and tokenizer...")
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("      Downloading T5-small (may take 1-2 min on first run)...")
    tokenizer = T5Tokenizer.from_pretrained("google/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/t5-small")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Model parameters: {total_params:,}")
    print("      ✓ Transformers OK")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# 3. Check Datasets
print("\n[3/6] Loading SuperGLUE CB dataset...")
try:
    from datasets import load_dataset
    print("      Loading 10 examples...")
    dataset = load_dataset("super_glue", "cb", split="train[:10]")
    print(f"      Loaded {len(dataset)} examples")
    print(f"      Example: {dataset[0]['premise'][:50]}...")
    print("      ✓ Datasets OK")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# 4. Check Tokenization
print("\n[4/6] Testing tokenization...")
try:
    text = "premise: The sky is blue. hypothesis: The sky is blue."
    tokens = tokenizer(text, return_tensors="pt")
    print(f"      Input IDs shape: {tokens['input_ids'].shape}")
    print(f"      Attention mask shape: {tokens['attention_mask'].shape}")
    print("      ✓ Tokenization OK")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# 5. Check PEFT (Prompt Tuning)
print("\n[5/6] Testing PEFT (Prompt Tuning)...")
try:
    from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
    
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=20,
        tokenizer_name_or_path="google/t5-small",
    )
    
    peft_model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    pct = 100 * trainable / total
    
    print(f"      Trainable parameters: {trainable:,}")
    print(f"      Total parameters: {total:,}")
    print(f"      Trainable %: {pct:.6f}%")
    print("      ✓ PEFT OK")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# 6. Check Forward Pass
print("\n[6/6] Testing forward pass...")
try:
    input_ids = tokenizer("premise: X hypothesis: Y", return_tensors="pt")['input_ids']
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    print(f"      Output logits shape: {outputs.logits.shape}")
    print("      ✓ Forward pass OK")
except Exception as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL CHECKS PASSED! Code is working correctly!")
print("="*80)
print("\nNext steps:")
print("1. Run VARIANT 2 for full training test (3-5 min)")
print("2. If successful, upload to server and run full experiment")
print("="*80)
