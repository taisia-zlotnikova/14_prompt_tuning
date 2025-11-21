# validate.py

import torch
from tqdm import tqdm
from utils import prepare_batch

def validate_model(prompt_model, val_loader, tokenizer, desc="Validation"):
    prompt_model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    counts = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            input_ids, attention_mask, labels = prepare_batch(batch)

            outputs = prompt_model(input_ids, attention_mask, labels)
            total_val_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            for i in range(len(labels)):
                # Берем только последний токен из predictions

                last_pred_token = predictions[i][-1].unsqueeze(0)  # сохраняем размерность
                last_true_token = labels[i][-1].unsqueeze(0)
                # last_pred_token = predictions[i][0].unsqueeze(0)  # сохраняем размерность
                # last_true_token = labels[i][0].unsqueeze(0) 

                # Заменяем -100 если нужно
                if last_true_token == -100:
                    last_true_token = torch.tensor([tokenizer.pad_token_id])
                
                # Декодируем только один токен
                pred_text = tokenizer.decode(last_pred_token, skip_special_tokens=False)
                true_text = tokenizer.decode(last_true_token, skip_special_tokens=False)

                if pred_text not in counts:
                    counts[pred_text] = 0
                counts[pred_text] += 1

                # print(f'pred_text = {pred_text}, true_text = {true_text}')

                if pred_text == true_text:
                    correct_predictions += 1
                total_predictions += 1

    print(f'counts = {counts}')
    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    metrics = {
        "loss": avg_val_loss,
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }

    return metrics

def print_metrics(metrics, phase="Validation"):
    """
    Красиво печатает метрики
    """
    print(f"\n📊 {phase} Metrics:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_predictions']})")