# validate.py

import torch
from tqdm import tqdm
from collections import Counter

def decode_answer(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return 1
    if text.startswith("no"):
        return 0
    return 0  # fallback

def validate_model(prompt_model, loader, tokenizer, desc="Validation"):
    prompt_model.eval()
    total = 0
    correct = 0

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids = batch["input_ids"].to(prompt_model.soft_prompt.device)
            attention_mask = batch["attention_mask"].to(prompt_model.soft_prompt.device)
            labels = batch["labels"].to(prompt_model.soft_prompt.device)

            # Generate answer
            outputs = prompt_model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=4
            )

            text = [tokenizer.decode(o) for o in outputs]
            preds = [decode_answer(tokenizer.decode(o)) for o in outputs]
            all_preds.extend(preds)

            # get true labels
            true_ids = labels[:, 0].tolist()   # first token of label (yes/no)
            true_texts = [decode_answer(tokenizer.decode([tid])) for tid in true_ids]

            break

            for p, t in zip(preds, true_texts):
                total += 1
                if p == t:
                    correct += 1
    print(Counter(all_preds))
    acc = correct / total
    return {"accuracy": acc}

def print_metrics(metrics, phase="Validation"):
    print(f"{phase} accuracy: {metrics['accuracy']:.4f}")
