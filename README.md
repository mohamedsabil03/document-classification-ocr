# AI Document OCR Classification & REST API Pipeline

An end-to-end Machine Learning pipeline and production-ready REST API for classifying document OCR text scans into multiple document categories (Invoices, Receipts, Resumes, Letters, Scientific Reports, Legal Contracts) using fine-tuned Hugging Face Transformer models (**DistilBERT** & **TinyBERT**).

---

## Project Overview

This repository implements a complete document classification lifecycle:
1. **Synthetic OCR Data Generation**: Realistic text templates with simulated OCR noise (character replacements, swaps, drops, spacing, and line break artifacts).
2. **Transformer Model Training**: Fine-tuning lightweight transformer models (**DistilBERT** and **TinyBERT**) via PyTorch.
3. **Model Evaluation & Selection**: Benchmark evaluation comparing Accuracy, F1-Score, Inference Latency (ms), Parameter Count, and Storage Size.
4. **FastAPI REST Service & Dashboard**: Asynchronous REST API serving predictions, featuring dynamic model switching and an embedded modern HTML5/CSS3 web dashboard.
5. **Automated Testing Suite**: Integration tests covering all API endpoints and validation edge cases.

---

## Project Architecture & Directory Structure

```
problem 1/
├── dataset/                    # Generated OCR datasets & label mappings
│   ├── train.csv               # 4,200 training samples (70%)
│   ├── val.csv                 # 900 validation samples (15%)
│   ├── test.csv                # 900 test samples (15%)
│   └── label_map.json          # Class label mapping dictionary
├── models/                     # Base Hugging Face model checkpoints
├── saved_models/               # Fine-tuned model outputs & evaluation summaries
│   ├── distilbert_doc_classifier/ # Saved DistilBERT model & tokenizer
│   ├── tinybert_doc_classifier/   # Saved TinyBERT model & tokenizer
│   ├── best_model_info.json    # Metadata on selected best model
│   ├── metrics_summary.json    # Detailed test metrics & confusion matrices
│   └── model_comparison.md     # Auto-generated markdown evaluation report
├── api.py                      # FastAPI REST server & API endpoints
├── index.html                  # Standalone Web Dashboard UI template
├── evaluate.py                 # Comprehensive model evaluation script
├── generate_dataset.py         # Synthetic OCR document text & noise generator
├── test_api.py                 # FastAPI endpoint test client script
├── train.py                    # PyTorch fine-tuning workflow script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Supported Document Classes

The system classifies document text into **6 distinct document categories**:
* `invoice`: Corporate billing statements, invoice numbers, line items, and totals.
* `receipt`: Store retail sales receipts with itemized items, subtotal, and tax.
* `resume`: Professional CVs with work history, education, skills, and contact info.
* `letter`: Formal business communications and correspondence.
* `scientific_report`: Academic paper abstracts, methodology, results, and citations.
* `legal_contract`: Non-disclosure agreements (NDAs), clauses, and legal signatures.

---

## ⚡ Key Features & Implementation Details

### 1. Synthetic Data Generator (`generate_dataset.py`)
* Generates **6,000 document samples** across the 6 document classes.
* **OCR Noise Injection**: Applies synthetic noise (random character replacement with OCR artifacts `|`, `]`, `[`, `_`, `-`, `~`, `0`, `1`, `l`, `I`, character swapping, deletion, and irregular line spacing) to **70% of dataset samples** to mimic real-world OCR scanning imperfections.
* Splits data into **70% Train (4,200)**, **15% Validation (900)**, and **15% Test (900)** sets.

### 2. Multi-Model Fine-Tuning (`train.py`)
* **DistilBERT** (`distilbert-base-uncased`): 66.9M parameters.
* **TinyBERT** (`TinyBERT_General_4L_312D`): 14.3M parameters.
* PyTorch `DataLoader` with custom `OCRDataset` handling truncation and padding (max length: 128 tokens).
* `AdamW` optimizer with linear learning rate warmup and gradient clipping.
* Best model checkpoint selection based on validation accuracy.

### 3. Comprehensive Evaluation Engine (`evaluate.py`)
* Evaluates models on 900 unseen test samples.
* Computes Accuracy, Weighted/Macro Precision, Recall, F1-Score, Per-Class Classification Reports, and Confusion Matrices.
* Measures CPU inference latency (ms per document) and model disk footprint (MB).

### 4. Production REST API & Web Dashboard (`api.py` & `index.html`)
* Built with **FastAPI** and **Uvicorn**.
* Pre-loads fine-tuned models at startup with zero-downtime model selection per request.
* **Dynamic Dataset Sampling (`GET /sample/{doc_type}`)**: Clicking any document category button in the UI dynamically fetches a fresh **random sample from the dataset**.
* **Side-by-Side Model Comparison**: Select `Compare Both Models` mode to evaluate DistilBERT and TinyBERT simultaneously, comparing predictions, confidence, and latencies live.
* **Standalone Web Dashboard (`index.html`)**: Served at `http://127.0.0.1:8000/`.
* **OpenAPI Documentation**: Automatically generated interactive Swagger UI (`/docs`).

### 5. Automated Endpoint Testing (`test_api.py`)
* Unit & integration tests utilizing `fastapi.testclient.TestClient`.
* Validates status codes, response structure, sample fetching, error handling (empty text, invalid model names), and classification accuracy across models.

---

## Model Performance Benchmark

Evaluated on 900 test samples across 6 document classes:

| Metric | DistilBERT (`distilbert-base-uncased`) | TinyBERT (`TinyBERT_General_4L_312D`) | Comparison |
| :--- | :--- | :--- | :--- |
| **Parameters** | 66,958,086 | 14,352,126 | TinyBERT is ~4.6x smaller |
| **Model Disk Size** | 256.12 MB | 55.44 MB | TinyBERT requires ~78% less disk space |
| **Avg CPU Latency** | ~63.23 ms/doc | ~11.60 ms/doc | TinyBERT is ~5.4x faster |
| **Accuracy** | **1.0000** | **1.0000** | Both achieve 100% test accuracy |
| **Precision (Weighted)**| **1.0000** | **1.0000** | Both achieve 1.0000 |
| **Recall (Weighted)** | **1.0000** | **1.0000** | Both achieve 1.0000 |
| **F1-Score (Weighted)** | **1.0000** | **1.0000** | Both achieve 1.0000 |

---

## Getting Started & Usage

### 1. Requirements & Installation

Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `torch`, `transformers`, `pandas`, `scikit-learn`)*

---

### 2. Generate Dataset

Generate 6,000 synthetic OCR document samples:
```bash
python generate_dataset.py
```
*Outputs: `dataset/train.csv`, `dataset/val.csv`, `dataset/test.csv`, `dataset/label_map.json`.*

---

### 3. Fine-Tune Models

Train both DistilBERT and TinyBERT models:
```bash
python train.py
```
*Saved Checkpoints: `saved_models/distilbert_doc_classifier/` & `saved_models/tinybert_doc_classifier/`.*

---

### 4. Evaluate Models

Benchmark models on the test set and select the winner:
```bash
python evaluate.py
```
*Outputs: `saved_models/metrics_summary.json`, `saved_models/best_model_info.json`, `saved_models/model_comparison.md`.*

---

### 5. Launch FastAPI Web Application & API

Run the production FastAPI server:
```bash
uvicorn api:app --reload --port 8000
```
* Web Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* Interactive API Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 6. Run API Test Suite

Validate the API endpoints with the test runner:
```bash
python test_api.py
```

---

## 🔌 API Endpoint Usage Examples

### `POST /predict`

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "INVOICE\nAcme Corp\nInvoice Number: INV-2026-9481\nTOTAL DUE: $1,450.00",
           "model_name": "distilbert"
         }'
```

**Response:**
```json
{
  "predicted_document_type": "invoice",
  "confidence_score": 0.9998,
  "model_used": "DistilBERT",
  "inference_time_ms": 14.32,
  "class_probabilities": {
    "invoice": 0.9998,
    "receipt": 0.0001,
    "resume": 0.0000,
    "letter": 0.0000,
    "scientific_report": 0.0000,
    "legal_contract": 0.0001
  }
}
```

### `GET /sample/{doc_type}`

Fetch a random document OCR text sample directly from the dataset (`invoice`, `receipt`, `resume`, `letter`, `scientific`, `contract`, or `random`).

**Request:**
```bash
curl -X GET "http://127.0.0.1:8000/sample/invoice"
```

**Response:**
```json
{
  "doc_type": "invoice",
  "label": "invoice",
  "text": "INVOICE\nApex Solutions\nInvoice Number: INV-2026-8381...",
  "source": "dataset"
}
```

---

## License

This project is open-source under the MIT License.

