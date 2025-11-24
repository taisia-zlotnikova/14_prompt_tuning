# PromptTuning - класс, реализующий prompt tuning, обучает одну матрицу - soft_prompt
import torch
import torch.nn as nn


class PromptTuningWrapper(nn.Module):
    def __init__(self, model, soft_prompt_length, hidden_dim, tokenizer, init_prompt=""):
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
        # if init_prompt is None:
        #     self.soft_prompt = nn.Parameter(
        #         torch.randn(soft_prompt_length, hidden_dim) * 0.02
        #     )
        # else:


    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        
        # Получаем эмбеддинги реального ввода
        input_embeds = self.model.shared(input_ids)  # или get_input_embeddings()

        # Создаём soft prompt эмбеддинги
        soft_prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Конкатенируем: [soft_prompt] + [input]
        full_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)

        # Расширяем attention mask
        prompt_mask = torch.ones(batch_size, self.soft_prompt_length, dtype=torch.long, device=input_ids.device)
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=labels
        )
        return outputs