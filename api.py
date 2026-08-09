import os
import json
import time
import csv
import random
from typing import Optional
from contextlib import asynccontextmanager
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SAVED_MODELS_DIR = "saved_models"
BEST_MODEL_INFO = os.path.join(SAVED_MODELS_DIR, "best_model_info.json")
LABEL_MAP_PATH = os.path.join(SAVED_MODELS_DIR, "label_map.json")
INDEX_HTML_PATH = "index.html"

# Model Registry Dictionary
LOADED_MODELS = {}
active_model_key = "distilbert"
label2id = {}
id2label = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Samples Cache
DATASET_SAMPLES = {}

def load_dataset_samples():
    global DATASET_SAMPLES
    DATASET_SAMPLES = {}
    
    dataset_dir = "dataset"
    for csv_file in ["val.csv", "test.csv", "train.csv"]:
        csv_path = os.path.join(dataset_dir, csv_file)
        if os.path.exists(csv_path):
            try:
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("text", "").strip()
                        label = row.get("label", "").strip()
                        if text and label:
                            if label not in DATASET_SAMPLES:
                                DATASET_SAMPLES[label] = []
                            DATASET_SAMPLES[label].append(text)
                if DATASET_SAMPLES:
                    total_samples = sum(len(v) for v in DATASET_SAMPLES.values())
                    print(f"Loaded {total_samples} dataset samples across {len(DATASET_SAMPLES)} classes from {csv_path}")
                    break
            except Exception as e:
                print(f"Warning loading dataset samples from {csv_path}: {e}")

    if not DATASET_SAMPLES:
        try:
            import generate_dataset
            for cls_name, gen_fn in generate_dataset.GENERATORS.items():
                DATASET_SAMPLES[cls_name] = [gen_fn() for _ in range(50)]
            print("Generated dynamic dataset samples via generate_dataset module.")
        except Exception as e:
            print(f"Could not load dynamic dataset generator: {e}")

def load_all_models():
    global label2id, id2label, active_model_key
    
    print("Initializing models and tokenizers for API service...")
    
    if not os.path.exists(LABEL_MAP_PATH):
        raise RuntimeError(f"Label map file not found at {LABEL_MAP_PATH}. Ensure models are trained first.")
    
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    # Load DistilBERT if available
    distil_dir = os.path.join(SAVED_MODELS_DIR, "distilbert_doc_classifier")
    if os.path.exists(distil_dir):
        print(f"Loading DistilBERT from: {distil_dir}")
        tok = AutoTokenizer.from_pretrained(distil_dir)
        mod = AutoModelForSequenceClassification.from_pretrained(distil_dir).to(device)
        mod.eval()
        LOADED_MODELS["distilbert"] = {
            "name": "DistilBERT",
            "dir": distil_dir,
            "tokenizer": tok,
            "model": mod,
            "params": "66.9M",
            "size": "256 MB",
            "avg_latency": "~65 ms"
        }

    # Load TinyBERT if available
    tiny_dir = os.path.join(SAVED_MODELS_DIR, "tinybert_doc_classifier")
    if os.path.exists(tiny_dir):
        print(f"Loading TinyBERT from: {tiny_dir}")
        tok = AutoTokenizer.from_pretrained(tiny_dir)
        mod = AutoModelForSequenceClassification.from_pretrained(tiny_dir).to(device)
        mod.eval()
        LOADED_MODELS["tinybert"] = {
            "name": "TinyBERT",
            "dir": tiny_dir,
            "tokenizer": tok,
            "model": mod,
            "params": "14.3M",
            "size": "55 MB",
            "avg_latency": "~15 ms"
        }

    if not LOADED_MODELS:
        raise RuntimeError("No fine-tuned models found in saved_models/")

    # Determine default active model
    if os.path.exists(BEST_MODEL_INFO):
        with open(BEST_MODEL_INFO, "r") as f:
            info = json.load(f)
        best_name = info.get("best_model_name", "").lower()
        if "tiny" in best_name and "tinybert" in LOADED_MODELS:
            active_model_key = "tinybert"
        elif "distil" in best_name and "distilbert" in LOADED_MODELS:
            active_model_key = "distilbert"

    print(f"API initialization complete. Active default model: {LOADED_MODELS[active_model_key]['name']}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    load_dataset_samples()
    yield

app = FastAPI(
    title="Document OCR Classification API",
    description="REST API & Dashboard serving fine-tuned lightweight Hugging Face transformer models for document OCR classification.",
    version="1.2.0",
    lifespan=lifespan
)

# Input schemas
class DocumentRequest(BaseModel):
    text: str = Field(
        ..., 
        description="Raw OCR text extracted from document scan.",
        json_schema_extra={"example": "INVOICE\nAcme Corp\nInvoice Number: INV-2026-9481\nTOTAL DUE: $1,450.00"}
    )
    model_name: Optional[str] = Field(
        "distilbert",
        description="Specify model to use: 'distilbert' or 'tinybert'. Default: 'distilbert'",
        json_schema_extra={"example": "distilbert"}
    )

# Output schema
class PredictionResponse(BaseModel):
    predicted_document_type: str
    confidence_score: float
    model_used: str
    inference_time_ms: float
    class_probabilities: dict

class SampleResponse(BaseModel):
    doc_type: str
    label: str
    text: str
    source: str

DOC_TYPE_ALIAS_MAP = {
    "invoice": "invoice",
    "receipt": "receipt",
    "resume": "resume",
    "letter": "letter",
    "scientific": "scientific_report",
    "scientific_report": "scientific_report",
    "contract": "legal_contract",
    "legal_contract": "legal_contract"
}

@app.get("/sample/{doc_type}", response_model=SampleResponse, tags=["Dataset Samples"])
@app.get("/sample", response_model=SampleResponse, tags=["Dataset Samples"])
def get_dataset_sample(doc_type: str = "random"):
    key = doc_type.lower().strip()
    
    if key == "random" or key not in DOC_TYPE_ALIAS_MAP:
        available_labels = list(DATASET_SAMPLES.keys()) if DATASET_SAMPLES else list(DOC_TYPE_ALIAS_MAP.values())
        target_label = random.choice(available_labels) if available_labels else "invoice"
    else:
        target_label = DOC_TYPE_ALIAS_MAP[key]

    if target_label in DATASET_SAMPLES and DATASET_SAMPLES[target_label]:
        sample_text = random.choice(DATASET_SAMPLES[target_label])
        source = "dataset"
    else:
        try:
            import generate_dataset
            gen_fn = generate_dataset.GENERATORS.get(target_label)
            if gen_fn:
                sample_text = gen_fn()
                source = "dynamic_generator"
            else:
                sample_text = f"Sample document text for {target_label}"
                source = "fallback"
        except Exception:
            sample_text = f"Sample document text for {target_label}"
            source = "fallback"

    return SampleResponse(
        doc_type=doc_type,
        label=target_label,
        text=sample_text,
        source=source
    )

@app.get("/", tags=["Dashboard"])
def render_dashboard():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse(content="<h1>Dashboard template index.html not found.</h1>", status_code=404)

@app.get("/info", tags=["Info"])
def api_info():
    return {
        "title": "Document Classification API",
        "status": "online",
        "active_default_model": LOADED_MODELS[active_model_key]["name"],
        "available_models": [v["name"] for v in LOADED_MODELS.values()],
        "supported_classes": list(label2id.keys()),
        "docs_url": "http://127.0.0.1:8000/docs",
        "predict_endpoint": "http://127.0.0.1:8000/predict",
        "sample_endpoint": "http://127.0.0.1:8000/sample/{doc_type}"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "active_default_model": LOADED_MODELS[active_model_key]["name"],
        "loaded_models": list(LOADED_MODELS.keys()),
        "device": str(device)
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Classification"])
def predict_document_type(payload: DocumentRequest):
    raw_text = payload.text.strip() if payload.text else ""
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input: 'text' field cannot be empty or contain only whitespace."
        )

    selected_key = active_model_key
    if payload.model_name:
        key_req = payload.model_name.lower().strip()
        if key_req in LOADED_MODELS:
            selected_key = key_req
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_name '{payload.model_name}'. Available options: {list(LOADED_MODELS.keys())}"
            )

    model_entry = LOADED_MODELS[selected_key]
    model_obj = model_entry["model"]
    tokenizer_obj = model_entry["tokenizer"]
    disp_name = model_entry["name"]

    try:
        start_t = time.time()
        
        inputs = tokenizer_obj(
            raw_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model_obj(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        predicted_class = id2label.get(pred_id, "unknown")
        confidence = float(probs[pred_id])
        latency_ms = round((time.time() - start_t) * 1000.0, 2)

        class_probs = {id2label[i]: float(round(probs[i], 4)) for i in range(len(id2label))}

        return PredictionResponse(
            predicted_document_type=predicted_class,
            confidence_score=round(confidence, 4),
            model_used=disp_name,
            inference_time_ms=latency_ms,
            class_probabilities=class_probs
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing error: {str(e)}"
        )

@app.exception_handler(Exception)
def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected server error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
