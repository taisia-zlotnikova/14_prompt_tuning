# train2.py
import torch
from tqdm import tqdm
from utils import prepare_batch

from config import device

def make_train2(prompt_model, train_loader):
    num_epochs = 3                 # для T5-small + BoolQ + soft prompt обычно хватает 8–20 эпох
    total_steps = len(train_loader) * num_epochs

    optimizer = torch.optim.AdamW([prompt_model.soft_prompt], lr=0.001, weight_decay=0.01)

    # Самый стабильный scheduler для prompt tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,      # плавно упадёт до ~0 к концу обучения
        eta_min=0.00001            # минимальный lr (можно 0.001–0.05)
    )

    print(f"Начинаем обучение на {num_epochs} эпох с lr=0.5 → cosine decay")

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Эпоха {epoch}/{num_epochs} ===")
        
        # ←←←←←←←←←←←←←←←←←←←←←←←←←←←
        prompt_model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            input_ids, attention_mask, labels = prepare_batch(batch)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = prompt_model(input_ids, attention_mask, labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch} loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")