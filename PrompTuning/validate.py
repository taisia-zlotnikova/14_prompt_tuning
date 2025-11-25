# validate.py

import torch
from tqdm import tqdm
from utils import prepare_batch

def validate_model(prompt_model, val_loader, tokenizer, desc="Validation"):
    prompt_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    counts = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            input_ids, attention_mask, labels = prepare_batch(batch)

            outputs = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            # print(predictions)
            # print(labels)
            
            for i in range(len(labels)):
                # Маска значимых токенов (не -100)
                last = 0
                significant_mask = labels[i] != -100
                flag = True
                for j in range(len(significant_mask)):
                    if not significant_mask[j]:
                        break
                    last = j
                    if labels[i][j] != predictions[i][j]:
                        flag = False
                        break
                text = tokenizer.decode(predictions[i][:last + 1], skip_special_tokens=True)
                counts[text] = counts.get(text, 0) + 1

                correct += flag
                total += 1

    print(f'Prediction counts: {counts}')
    accuracy = correct / total if total > 0 else 0
    return accuracy  # ← ВОЗВРАЩАЕМ ЧИСЛО, а не словарь!

def print_metrics(metrics, phase="Validation"):
    """
    Красиво печатает метрики
    """
    print(f"\n📊 {phase} Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_predictions']})")

