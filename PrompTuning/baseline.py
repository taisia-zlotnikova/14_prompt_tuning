import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# utils.py

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


# load_data.py
from datasets import load_dataset
from torch.utils.data import DataLoader
def tokenize_element(element, tokenizer, max_length=128):
    enc = tokenizer(
        element["passage"],
        element["question"],
        truncation=True,
        padding="max_length",   # or "longest"
        max_length=256,
    )
    enc["labels"] = torch.full((256,), -100)
    label = "yes" if element["label"] == 1 else "no"
    enc["labels"][255] = torch.tensor(tokenizer([label])["input_ids"][0][0], dtype=torch.long)
    for k, v in enc.items():
        enc[k] = torch.tensor(v)
    return {"input_ids" : enc["input_ids"],
            "attention_mask" : enc["attention_mask"],
            "labels" : enc["labels"]}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = [tokenize_element(elem, tokenizer) for elem in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_superglue_task(task_name, tokenizer, batch_size=2, max_sizes={"train": None, "validation": None, "test": None}):
    dataset = load_dataset("super_glue", task_name)

    print(f"size dataset before decrease = {dataset.shape}")
    for name, size in max_sizes.items():
        if size is not None:
            dataset[name] = dataset[name].shuffle(seed=42).select(range(size))
    print(f"size dataset after decrease = {dataset.shape}")

    train_dataset = MyDataset(dataset["train"], tokenizer)
    val_dataset = MyDataset(dataset["validation"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) #, collate_fn=collate_fn)

    return train_loader, val_loader

# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# train_loader, val_loader = get_superglue_task("boolq", tokenizer)
# for batch in tqdm(train_loader, desc="Training"):
#     for k, v in batch.items():
#         print(k, v)
#         if k == "input_ids":
#             print(len(v))
#     break
torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PromptTuningWrapper.py
import torch.nn as nn

# Здесь предалагаю реализовать на торче prompt tuning
# PromptTuning - класс, реализующий prompt tuning, обучает одну матрицу - soft_prompt

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


# validate.py

def validate_model(prompt_model, val_loader, tokenizer, desc="Validation"):
    prompt_model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            input_ids, attention_mask, labels = prepare_batch(batch)

            outputs = prompt_model(input_ids, attention_mask, labels)
            total_val_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            for i in range(len(labels)):
                # Берем только последний токен из predictions

                last_pred_token = predictions[i][-1].unsqueeze(0)  # сохраняем размерность
                last_true_token = labels[i][-1].unsqueeze(0)
                
                # Заменяем -100 если нужно
                if last_true_token == -100:
                    last_true_token = torch.tensor([tokenizer.pad_token_id])
                
                # Декодируем только один токен
                pred_text = tokenizer.decode(last_pred_token, skip_special_tokens=False)
                true_text = tokenizer.decode(last_true_token, skip_special_tokens=False)

                if pred_text == true_text:
                    correct_predictions += 1
                total_predictions += 1

            if total_predictions % 1000 == 0:
                print(correct_predictions, total_predictions, 1.0 * correct_predictions / total_predictions)

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    metrics = {
        "loss": avg_val_loss,
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }

    return metrics

def print_metrics(metrics, phase="Validation"):
    """
    Красиво печатает метрики
    """
    print(f"\n📊 {phase} Metrics:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_predictions']})")
# train.py

# Глобальная модель
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
for p in model.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Создаём soft prompt wrapper
prompt_model = PromptTuningWrapper(model, soft_prompt_length=20, hidden_dim=model.config.hidden_size).to(device)

# Загружаем датасет
max_sizes = {"train": 1000, "validation": None, "test": 50}
train_loader, val_loader = get_superglue_task("boolq", tokenizer, max_sizes=max_sizes)

# Оптимизатор
optimizer = torch.optim.Adam([prompt_model.soft_prompt], lr=1e-3)
def make_train():
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
loss = make_train()
print(loss)
print("\nОбучение завершено!")
print("\nНачинаем валидацию\n")
metrics = validate_model(prompt_model, val_loader, tokenizer, desc="Validation")
print_metrics(metrics)