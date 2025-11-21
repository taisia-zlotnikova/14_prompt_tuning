# train.py
import torch
from tqdm import tqdm
from utils import prepare_batch

def make_train(prompt_model, train_loader, optimizer):
    prompt_model.train()
    total_loss = 0.
    num_batches = 0
    grad_norms = []
    lr_changes = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, labels = prepare_batch(batch)
        # forward
        outputs = prompt_model(input_ids, attention_mask, labels)
        loss = outputs.loss

        loss.backward()
        grad_norm = torch.norm(prompt_model.soft_prompt.grad).item()
        grad_norms.append(grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach().item()
        num_batches += 1


        # ----------------------------
        current_lr = optimizer.param_groups[0]['lr']
        
        if grad_norm < 1e-6:  # Слишком маленькие градиенты
            new_lr = min(current_lr * 2.0, 1.0)  # Увеличиваем LR, но не больше 1.0
            print(f"  Батч {num_batches}: Увеличиваем LR {current_lr:.6f} → {new_lr:.6f} (grad_norm={grad_norm:.6f})")
            optimizer.param_groups[0]['lr'] = new_lr
            lr_changes += 1
                
        elif grad_norm > 100.0:  # Слишком большие градиенты  
            new_lr = max(current_lr * 0.8, 1e-6)  # Уменьшаем LR, но не меньше 1e-6
            print(f"  Батч {num_batches}: Уменьшаем LR {current_lr:.6f} → {new_lr:.6f} (grad_norm={grad_norm:.6f})")
            optimizer.param_groups[0]['lr'] = new_lr
            lr_changes += 1
        # -----------------------------

        # if num_batches % 5 == 0:  # каждые 5 батчей
        #     current_lr = optimizer.param_groups[0]['lr']
        #     print(f"Батч {num_batches}:")
        #     print(f"  loss: {loss.item():.4f}")
        #     print(f"  grad_norm: {grad_norm:.6f}")
            
        #     # КРИТИЧЕСКАЯ ДИАГНОСТИКА:
        #     if grad_norm < 1e-6:
        #         print("  🚨 СЛИШКОМ МАЛЫЕ ГРАДИЕНТЫ! Увеличьте LR")
        #     elif grad_norm > 5.0:
        #         print("  🚨 СЛИШКОМ БОЛЬШИЕ ГРАДИЕНТЫ! Уменьшите LR")

    return total_loss / num_batches

