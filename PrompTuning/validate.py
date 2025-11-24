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

            outputs = prompt_model(input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            # print("predictions = ", predictions)
            # print("labels = ", labels)
            
            for i in range(len(labels)):
                # Берем первый токен предсказания (игнорируя специальные токены)
                # В T5 первый токен обычно и есть ответ
                pred_token = predictions[i][0]  # Первый токен
                true_token = labels[i][0]       # Первый токен цели
                
                # Пропускаем специальные токены (если нужно)
                if pred_token in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
                    continue
                    
                pred_text = tokenizer.decode(pred_token, skip_special_tokens=True)
                true_text = tokenizer.decode(true_token, skip_special_tokens=True)
                
                counts[pred_text] = counts.get(pred_text, 0) + 1
                
                if pred_token == true_token:
                    correct += 1
                total += 1

    print(f'Prediction counts: {counts}')
    return {
        "loss": total_loss / len(val_loader),
        "accuracy": correct / total if total > 0 else 0,
        "correct_predictions": correct,
        "total_predictions": total
    }

def print_metrics(metrics, phase="Validation"):
    """
    Красиво печатает метрики
    """
    print(f"\n📊 {phase} Metrics:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_predictions']})")

