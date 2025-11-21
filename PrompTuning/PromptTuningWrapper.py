# PromptTuningWrapper.py

import torch
import torch.nn as nn

class PromptTuningWrapper(nn.Module):
    def __init__(self, model, soft_prompt_length, hidden_dim):
        super().__init__()

        self.model = model
        self.soft_prompt_length = soft_prompt_length
        self.hidden_dim = hidden_dim

        for p in self.model.parameters():
            p.requires_grad = False

        self.soft_prompt = nn.Parameter(
            torch.randn(soft_prompt_length, hidden_dim) * 0.02
        )

    def forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.size(0)

        # Input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Expand prompt
        expanded_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Prepend
        full_embeds = torch.cat([expanded_prompt, input_embeds], dim=1)

        # Attention mask
        prompt_mask = torch.ones(batch_size, self.soft_prompt_length, device=attention_mask.device)
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Labels (pad left side with -100)
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.soft_prompt_length), -100, device=labels.device
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        return self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels
        )
