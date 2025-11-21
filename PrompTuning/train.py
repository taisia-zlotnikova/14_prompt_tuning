# train.py
import torch
from tqdm import tqdm
from utils import prepare_batch

def make_train(prompt_model, train_loader, lr = 1e-3):
    optimizer = torch.optim.AdamW(
            [prompt_model.soft_prompt],
            lr=lr,
            weight_decay=0.01
        )
    prompt_model.train()
    total_loss = 0.
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, labels = prepare_batch(batch)
        # forward
        outputs = prompt_model(input_ids, attention_mask, labels)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach().item()
        num_batches += 1

    return total_loss / num_batches

