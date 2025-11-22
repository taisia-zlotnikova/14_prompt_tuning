import torch
from tqdm import tqdm
from utils import prepare_batch

def clean_token(text):
    text = text.replace("▁", "")
    text = text.replace("<pad>", "")
    text = text.replace("</s>", "")
    return text.strip().lower()

def extract_first_label_token(label_ids, tokenizer):
    """
    Находим первый НЕ -100 и НЕ <pad> токен в label.
    """
    pad = tokenizer.pad_token_id

    for tok in label_ids:
        if tok != -100 and tok != pad and tok != 1 and tok != 3 and tok != 0:
            return tok

    # fallback — крайне редко
    return pad


def validate_model(prompt_model, val_loader, tokenizer, desc="Validation"):
    prompt_model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    freq = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            input_ids, attention_mask, labels = prepare_batch(batch)

            outputs = prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

            # [B, T, V]
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for i in range(len(labels)):

                # --- TRUE ---
                true_tok = extract_first_label_token(labels[i], tokenizer)
                true_tok = tokenizer.decode([true_tok])
                # true_text = clean_token(true_tok)
                true_text = true_tok

                # --- PRED ---
                # print(predictions[i])
                pred_tok = extract_first_label_token(predictions[i], tokenizer)   # первое предсказание decoder-a
                pred_tok = tokenizer.decode([pred_tok])
                pred_text = pred_tok
                # pred_text = clean_token(pred_tok)

                # print(true_text, pred_text)

                # считаем частоту
                freq[pred_text] = freq.get(pred_text, 0) + 1

                # бинарное сравнение
                if true_text in ["yes", "no"] and pred_text == true_text:
                    correct += 1

                total += 1

    avg_loss = total_val_loss / len(val_loader)
    acc = correct / total

    print("\nPrediction frequencies:", freq)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "correct_predictions": correct,
        "total_predictions": total
    }


def print_metrics(metrics, phase="Validation"):
    print(f"\n📊 {phase} Metrics:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct_predictions']}/{metrics['total_predictions']})")
