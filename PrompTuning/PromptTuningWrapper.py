# PromptTuning - класс, реализующий prompt tuning, обучает одну матрицу - soft_prompt
import torch
import torch.nn as nn


class PromptTuningWrapper(nn.Module):
    def __init__(self, model, soft_prompt_length, hidden_dim, tokenizer):
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

    # def __init__(self, model, soft_prompt_length, hidden_dim, tokenizer, init_text=None):
    #     super().__init__()
        
    #     self.model = model
    #     self.soft_prompt_length = soft_prompt_length
    #     self.hidden_dim = hidden_dim
    #     self.tokenizer = tokenizer

    #     for p in self.model.parameters():
    #         p.requires_grad = False

    #     # Инициализация из реальных слов
    #     if init_text is None:
    #         # Хороший промпт для BoolQ
    #         init_text = "Answer the question based on the passage. Passage: {passage} Question: {question} Answer:"
        
    #     # Токенизируем и берем эмбеддинги
    #     init_tokens = tokenizer(
    #         init_text, 
    #         max_length=soft_prompt_length, 
    #         truncation=True, 
    #         return_tensors="pt"
    #     )
        
    #     init_ids = init_tokens["input_ids"].squeeze(0)
        
    #     # Берем эмбеддинги этих токенов как начальные значения
    #     with torch.no_grad():
    #         init_embeddings = model.get_input_embeddings()(init_ids)
        
    #     # Если текст короче, чем нужная длина - дополняем случайными векторами
    #     if init_embeddings.size(0) < soft_prompt_length:
    #         padding_length = soft_prompt_length - init_embeddings.size(0)
    #         padding = torch.randn(padding_length, hidden_dim) * 0.01
    #         init_embeddings = torch.cat([init_embeddings, padding], dim=0)
    #     elif init_embeddings.size(0) > soft_prompt_length:
    #         init_embeddings = init_embeddings[:soft_prompt_length]
        
    #     self.soft_prompt = nn.Parameter(init_embeddings)

    def forward(self, input_ids, attention_mask, labels=None):
        input_embeds = self.model.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)

        # Добавляем soft prompt
        expanded_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_embeds = torch.cat([expanded_prompt, input_embeds], dim=1)

        # Расширяем attention_mask
        prompt_mask = torch.ones(batch_size, self.soft_prompt_length, 
                                dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=labels  
        )
        return outputs