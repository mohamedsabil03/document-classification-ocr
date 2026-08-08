import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

MODEL_CONFIGS = [
    {
        "name": "DistilBERT",
        "path": "models/distilbert_base_uncased",
        "output_dir": os.path.join(SAVED_MODELS_DIR, "distilbert_doc_classifier"),
        "lr": 3e-5,
        "epochs": 3,
        "batch_size": 32,
    },
    {
        "name": "TinyBERT",
        "path": "models/tinybert_general_4l_312d",
        "output_dir": os.path.join(SAVED_MODELS_DIR, "tinybert_doc_classifier"),
        "lr": 5e-5,
        "epochs": 3,
        "batch_size": 32,
    }
]

class OCRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train_model(config, train_df, val_df, label2id, id2label):
    print(f"\n==================================================", flush=True)
    print(f" Starting Fine-Tuning for: {config['name']}", flush=True)
    print(f" Path: {config['path']}", flush=True)
    print(f"==================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["path"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    train_dataset = OCRDataset(train_df["text"].tolist(), [label2id[l] for l in train_df["label"]], tokenizer)
    val_dataset = OCRDataset(val_df["text"].tolist(), [label2id[l] for l in val_df["label"]], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            if batch_idx % 5 == 0 or batch_idx == len(train_loader):
                print(f"  [{config['name']}] Epoch {epoch}/{config['epochs']} | Batch {batch_idx}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total

        print(f"--> [{config['name']}] Epoch {epoch}/{config['epochs']} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", flush=True)

    training_time = time.time() - start_time
    print(f"Training for {config['name']} completed in {training_time:.2f} seconds.", flush=True)

    os.makedirs(config["output_dir"], exist_ok=True)
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    meta_info = {
        "model_name": config["name"],
        "base_model_path": config["path"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "training_time_seconds": round(training_time, 2),
        "final_val_acc": round(val_acc, 4)
    }
    with open(os.path.join(config["output_dir"], "training_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Successfully saved {config['name']} to: {config['output_dir']}", flush=True)

def main():
    with open("dataset/label_map.json", "r") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    with open(os.path.join(SAVED_MODELS_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/val.csv")

    for config in MODEL_CONFIGS:
        train_model(config, train_df, val_df, label2id, id2label)

if __name__ == "__main__":
    main()
