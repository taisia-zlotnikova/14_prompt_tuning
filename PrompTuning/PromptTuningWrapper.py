# PromptTuning - класс, реализующий prompt tuning, обучает одну матрицу - soft_prompt
import torch
import torch.nn as nn


class PromptTuningWrapper(nn.Module):
    def __init__(self, model, soft_prompt_length, hidden_dim):
        """
        model - базовая модель для prompt tuning
        soft_prompt_length — длина обучаемого эмбеддинга
        hidden_dim - размерность обучаемого эмбеддинга
        """
        super().__init__()

        self.model = model
        self.soft_prompt_length = soft_prompt_length
        self.hidden_dim = hidden_dim

        for p in self.model.parameters():
            p.requires_grad = False

        # make random sort prompt
        self.soft_prompt = nn.Parameter(
            torch.randn(soft_prompt_length, hidden_dim) * 0.02
        )

    def forward(self, input_ids, attention_mask, labels):
        # YOUR CODE BELOW - поменять input_ids, attention_mask с учетом soft prompt
        input_embeds = self.model.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)
        expanded_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_prompt_embeds = torch.cat([expanded_prompt, input_embeds], dim=1)

        prompt_mask = torch.ones(batch_size, self.soft_prompt_length, device=attention_mask.device)
        full_prompt_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if labels is not None:
            prompt_labels = torch.full((batch_size, self.soft_prompt_length), -100,
                                    device=labels.device)
            labels = torch.cat([prompt_labels, labels], dim=1)

        # код ниже можно оставить как есть
        outputs = self.model(
            inputs_embeds=full_prompt_embeds,
            attention_mask=full_prompt_mask,
            labels=labels
        )

        return outputs