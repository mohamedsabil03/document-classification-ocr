import os
import json
import time
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SAVED_MODELS_DIR = "saved_models"
TEST_FILE = "dataset/test.csv"
LABEL_MAP_FILE = os.path.join(SAVED_MODELS_DIR, "label_map.json")

def get_dir_size_mb(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)

def evaluate_model(model_name, model_dir, test_df, label2id, id2label, device):
    print(f"\nEvaluating {model_name} from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    texts = test_df["text"].tolist()
    true_labels = [label2id[l] for l in test_df["label"].tolist()]

    predictions = []
    probabilities = []
    latencies = []

    with torch.no_grad():
        for text in texts:
            start_t = time.time()
            inputs = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt").to(device)
            outputs = model(**inputs)
            latency = (time.time() - start_t) * 1000.0  # ms
            
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            
            predictions.append(pred_id)
            probabilities.append(probs)
            latencies.append(latency)

    acc = accuracy_score(true_labels, predictions)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted", zero_division=0)
    
    avg_latency = float(np.mean(latencies))
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = get_dir_size_mb(model_dir)

    target_names = [id2label[i] for i in range(len(id2label))]
    report_dict = classification_report(true_labels, predictions, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, predictions).tolist()

    results = {
        "model_name": model_name,
        "model_dir": model_dir,
        "parameters": param_count,
        "model_size_mb": model_size_mb,
        "avg_latency_ms": round(avg_latency, 2),
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class_metrics": {name: report_dict[name] for name in target_names},
        "confusion_matrix": cm
    }

    return results

def main():
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test dataset not found at {TEST_FILE}. Run generate_dataset.py first.")
    
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    test_df = pd.read_csv(TEST_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_eval = [
        ("DistilBERT", os.path.join(SAVED_MODELS_DIR, "distilbert_doc_classifier")),
        ("TinyBERT", os.path.join(SAVED_MODELS_DIR, "tinybert_doc_classifier"))
    ]

    all_metrics = {}
    for name, model_dir in models_to_eval:
        metrics = evaluate_model(name, model_dir, test_df, label2id, id2label, device)
        all_metrics[name] = metrics

    # Print Summary Table
    print("\n" + "="*80)
    print("                      MODEL PERFORMANCE EVALUATION")
    print("="*80)
    print(f"{'Metric':<25} | {'DistilBERT':<20} | {'TinyBERT':<20}")
    print("-" * 80)
    print(f"{'Parameters':<25} | {all_metrics['DistilBERT']['parameters']:<20,d} | {all_metrics['TinyBERT']['parameters']:<20,d}")
    print(f"{'Model Size (MB)':<25} | {all_metrics['DistilBERT']['model_size_mb']:<20.2f} | {all_metrics['TinyBERT']['model_size_mb']:<20.2f}")
    print(f"{'Avg Latency (ms)':<25} | {all_metrics['DistilBERT']['avg_latency_ms']:<20.2f} | {all_metrics['TinyBERT']['avg_latency_ms']:<20.2f}")
    print(f"{'Accuracy':<25} | {all_metrics['DistilBERT']['accuracy']:<20.4f} | {all_metrics['TinyBERT']['accuracy']:<20.4f}")
    print(f"{'Precision (Weighted)':<25} | {all_metrics['DistilBERT']['precision_weighted']:<20.4f} | {all_metrics['TinyBERT']['precision_weighted']:<20.4f}")
    print(f"{'Recall (Weighted)':<25} | {all_metrics['DistilBERT']['recall_weighted']:<20.4f} | {all_metrics['TinyBERT']['recall_weighted']:<20.4f}")
    print(f"{'F1-Score (Weighted)':<25} | {all_metrics['DistilBERT']['f1_weighted']:<20.4f} | {all_metrics['TinyBERT']['f1_weighted']:<20.4f}")
    print("="*80)

    # Determine Winner
    distil_score = all_metrics['DistilBERT']['f1_weighted']
    tiny_score = all_metrics['TinyBERT']['f1_weighted']
    
    if distil_score >= tiny_score:
        winner = "DistilBERT"
    else:
        winner = "TinyBERT"

    print(f"\nWinning Model Selected: {winner} (F1-Weighted: {all_metrics[winner]['f1_weighted']})")

    # Save metrics summary
    with open(os.path.join(SAVED_MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    best_info = {
        "best_model_name": winner,
        "best_model_dir": all_metrics[winner]["model_dir"],
        "metrics": all_metrics[winner]
    }
    with open(os.path.join(SAVED_MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    # Write Markdown Summary
    md_content = f"""# Model Performance Comparison Summary

Evaluated on held-out test dataset ({len(test_df)} OCR document samples across 6 document classes).

## Overall Metrics

| Metric | DistilBERT (`distilbert-base-uncased`) | TinyBERT (`TinyBERT_General_4L_312D`) | Better Model |
| :--- | :--- | :--- | :--- |
| **Parameters** | {all_metrics['DistilBERT']['parameters']:,} | {all_metrics['TinyBERT']['parameters']:,} | TinyBERT (~4.6x smaller) |
| **Model Disk Size** | {all_metrics['DistilBERT']['model_size_mb']} MB | {all_metrics['TinyBERT']['model_size_mb']} MB | TinyBERT |
| **Avg Latency (CPU)** | {all_metrics['DistilBERT']['avg_latency_ms']} ms/doc | {all_metrics['TinyBERT']['avg_latency_ms']} ms/doc | TinyBERT (~3-4x faster) |
| **Accuracy** | **{all_metrics['DistilBERT']['accuracy']:.4f}** | **{all_metrics['TinyBERT']['accuracy']:.4f}** | {winner} |
| **Precision (Weighted)** | **{all_metrics['DistilBERT']['precision_weighted']:.4f}** | **{all_metrics['TinyBERT']['precision_weighted']:.4f}** | {winner} |
| **Recall (Weighted)** | **{all_metrics['DistilBERT']['recall_weighted']:.4f}** | **{all_metrics['TinyBERT']['recall_weighted']:.4f}** | {winner} |
| **F1-Score (Weighted)** | **{all_metrics['DistilBERT']['f1_weighted']:.4f}** | **{all_metrics['TinyBERT']['f1_weighted']:.4f}** | {winner} |

## Recommendation & Deployment Choice

- **Selected Model for REST API**: **{winner}**
- **Location**: `{all_metrics[winner]['model_dir']}`

"""
    with open(os.path.join(SAVED_MODELS_DIR, "model_comparison.md"), "w") as f:
        f.write(md_content)

    print("Evaluation and report generation complete!")

if __name__ == "__main__":
    main()
