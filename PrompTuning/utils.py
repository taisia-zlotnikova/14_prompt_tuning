# utils.py
import torch
from config import device  # импортируем device из config

def expand_soft_prompt(soft_prompt, batch_size):
    # soft_prompt: [prompt_len, hidden_dim]
    # возвращаем [batch_size, prompt_len, hidden_dim]
    return soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

def extend_attention_mask(attention_mask, prompt_len):
    # attention_mask: [batch, seq_len]
    prompt_mask = torch.ones(attention_mask.size(0), prompt_len, device=attention_mask.device)
    return torch.cat([prompt_mask, attention_mask], dim=1)

def extend_labels(labels, prompt_len):
    pad_labels = torch.full((labels.size(0), prompt_len), -100, device=labels.device)
    return torch.cat([pad_labels, labels], dim=1)

def prepare_batch(batch):
    input_ids = torch.stack([item for item in batch["input_ids"]]).to(device)
    attention_mask = torch.stack([item for item in batch["attention_mask"]]).to(device)
    labels = torch.stack([item for item in batch["labels"]]).to(device)
    
    return input_ids, attention_mask, labels
