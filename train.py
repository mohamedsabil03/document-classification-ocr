import os
import json
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
SEED = 42

MODEL_CONFIGS = [
    {
        "name": "DistilBERT",
        "path": "models/distilbert_base_uncased" if os.path.exists("models/distilbert_base_uncased") else "distilbert-base-uncased",
        "output_dir": os.path.join(SAVED_MODELS_DIR, "distilbert_doc_classifier"),
        "lr": 3e-5,
        "epochs": 4,
        "batch_size": 32,
    },
    {
        "name": "TinyBERT",
        "path": "models/tinybert_general_4l_312d" if os.path.exists("models/tinybert_general_4l_312d") else "huawei-noah/TinyBERT_General_4L_312D",
        "output_dir": os.path.join(SAVED_MODELS_DIR, "tinybert_doc_classifier"),
        "lr": 5e-5,
        "epochs": 4,
        "batch_size": 32,
    }
]

class OCRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = "" if pd.isna(self.texts[idx]) else str(self.texts[idx])
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
    best_val_acc = float("-inf")
    best_epoch = 0
    os.makedirs(config["output_dir"], exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
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
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch["labels"]
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total

        print(f"--> [{config['name']}] Epoch {epoch}/{config['epochs']} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save_pretrained(config["output_dir"])
            tokenizer.save_pretrained(config["output_dir"])
            print(f"  Saved new best checkpoint (epoch {epoch}).", flush=True)

    training_time = time.time() - start_time
    print(f"Training for {config['name']} completed in {training_time:.2f} seconds.", flush=True)

    meta_info = {
        "model_name": config["name"],
        "base_model_path": config["path"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "training_time_seconds": round(training_time, 2),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4)
    }
    with open(os.path.join(config["output_dir"], "training_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Successfully saved {config['name']} to: {config['output_dir']}", flush=True)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    with open("dataset/label_map.json", "r") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    with open(os.path.join(SAVED_MODELS_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/val.csv")

    required_columns = {"text", "label"}
    for name, dataframe in (("train", train_df), ("validation", val_df)):
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing_columns)}")
        if dataframe.empty:
            raise ValueError(f"{name}.csv must contain at least one example.")
        unknown_labels = set(dataframe["label"]) - set(label2id)
        if unknown_labels:
            raise ValueError(f"{name}.csv has labels absent from label_map.json: {sorted(unknown_labels)}")

    for config in MODEL_CONFIGS:
        train_model(config, train_df, val_df, label2id, id2label)

if __name__ == "__main__":
    main()
