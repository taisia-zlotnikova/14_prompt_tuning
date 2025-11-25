# models.py

import torch
from transformers import AutoModelForSeq2SeqLM
from PromptTuningWrapper import PromptTuningWrapper
from peft import PromptTuningConfig, get_peft_model
from transformers import T5ForConditionalGeneration
from config import device

# DEVICE
torch.cuda.set_device(4)
torch.cuda.empty_cache()

name_model = "t5-small"

def my_model(tokenizer, prompt_length=20, init_prompt=''):
    """Кастомная модель с PromptTuningWrapper"""
    model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
        
    prompt_model = PromptTuningWrapper(
        model,
        soft_prompt_length=prompt_length,
        hidden_dim=model.config.hidden_size,
        tokenizer=tokenizer,
        init_prompt=init_prompt
    ).to(device)

    return prompt_model

def peft_model(tokenizer, prompt_length=20, init_prompt=''):
    """PEFT модель"""
    model = T5ForConditionalGeneration.from_pretrained(name_model)
    
    config = PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        num_virtual_tokens=prompt_length,
        prompt_tuning_init_text=init_prompt
    )
    prompt_model = get_peft_model(model, config).to(device)
    return prompt_model
