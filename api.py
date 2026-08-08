import os
import json
import time
from typing import Optional
from contextlib import asynccontextmanager
# Try importing torch & transformers (Local execution)
# Fallback to httpx Hugging Face Inference API (Vercel Serverless execution)
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    USE_LOCAL_TORCH = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    USE_LOCAL_TORCH = False
    import httpx
    device = "cpu (remote HF API)"



def load_all_models():
    global active_model_key, label2id, id2label

    print(f"Initializing models (Local Torch: {USE_LOCAL_TORCH})...")

    if USE_LOCAL_TORCH:
        for model_key, repo_id in MODEL_REPOS.items():
            try:
                print(f"Loading {model_key} locally from Hugging Face: {repo_id}")
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                model = AutoModelForSequenceClassification.from_pretrained(repo_id).to(device)
                model.eval()

                model_id2label = {int(k): v for k, v in model.config.id2label.items()}
                model_label2id = {v: k for k, v in model_id2label.items()}

                LOADED_MODELS[model_key] = {
                    "name": "DistilBERT" if model_key == "distilbert" else "TinyBERT",
                    "repo": repo_id,
                    "tokenizer": tokenizer,
                    "model": model,
                    "id2label": model_id2label,
                    "label2id": model_label2id,
                    "params": "66.9M" if model_key == "distilbert" else "14.3M",
                    "size": "256 MB" if model_key == "distilbert" else "55 MB",
                    "avg_latency": "~65 ms" if model_key == "distilbert" else "~15 ms"
                }
                print(f"Successfully loaded {model_key} locally")
            except Exception as e:
                print(f"ERROR loading {model_key}: {str(e)}")
    else:
        import httpx
        for model_key, repo_id in MODEL_REPOS.items():
            try:
                print(f"Registering {model_key} via Hugging Face Serverless API: {repo_id}")
                model_id2label = {
                    0: "invoice",
                    1: "receipt",
                    2: "resume",
                    3: "letter",
                    4: "scientific_report",
                    5: "legal_contract"
                }
                try:
                    resp = httpx.get(f"https://huggingface.co/{repo_id}/raw/main/config.json", timeout=5.0)
                    if resp.status_code == 200:
                        cfg = resp.json()
                        if "id2label" in cfg:
                            model_id2label = {int(k): v for k, v in cfg["id2label"].items()}
                except Exception:
                    pass

                model_label2id = {v: k for k, v in model_id2label.items()}

                LOADED_MODELS[model_key] = {
                    "name": "DistilBERT" if model_key == "distilbert" else "TinyBERT",
                    "repo": repo_id,
                    "id2label": model_id2label,
                    "label2id": model_label2id,
                    "params": "66.9M" if model_key == "distilbert" else "14.3M",
                    "size": "256 MB" if model_key == "distilbert" else "55 MB",
                    "avg_latency": "~65 ms" if model_key == "distilbert" else "~15 ms"
                }
                print(f"Successfully registered {model_key} for HF API inference")
            except Exception as e:
                print(f"ERROR registering {model_key}: {str(e)}")

    if not LOADED_MODELS:
        raise RuntimeError("Could not load or register any models.")

    if "distilbert" in LOADED_MODELS:
        active_model_key = "distilbert"
    elif "tinybert" in LOADED_MODELS:
        active_model_key = "tinybert"

    id2label = LOADED_MODELS[active_model_key]["id2label"]
    label2id = LOADED_MODELS[active_model_key]["label2id"]

    print("API initialization complete.")
    print(f"Active default model: {LOADED_MODELS[active_model_key]['name']}")
    print(f"Loaded models: {list(LOADED_MODELS.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
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

class ModelSwitchRequest(BaseModel):
    model_name: str = Field(
        ...,
        description="Model key to set as active default: 'distilbert' or 'tinybert'",
        json_schema_extra={"example": "tinybert"}
    )

# Output schema
class PredictionResponse(BaseModel):
    predicted_document_type: str
    confidence_score: float
    model_used: str
    inference_time_ms: float
    class_probabilities: dict

# Embedded Dashboard HTML
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Document Classification Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent-blue: #38bdf8;
            --accent-purple: #a855f7;
            --accent-green: #34d399;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 24px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto 28px auto;
            width: 100%;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3);
        }

        .brand-title h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #f8fafc, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-title p {
            font-size: 13px;
            color: var(--text-muted);
        }

        .badge-status {
            background: rgba(52, 211, 153, 0.1);
            border: 1px solid rgba(52, 211, 153, 0.3);
            color: var(--accent-green);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge-status span {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .templates-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
        }

        .template-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .template-btn:hover {
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--accent-blue);
            color: var(--text-main);
        }

        textarea {
            width: 100%;
            height: 220px;
            background-color: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            color: var(--text-main);
            padding: 14px;
            font-size: 13.5px;
            line-height: 1.5;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s ease;
        }

        textarea:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
        }

        .controls-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 16px;
            gap: 16px;
        }

        .select-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .select-group label {
            font-size: 13px;
            color: var(--text-muted);
        }

        select {
            background-color: #0f172a;
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            cursor: pointer;
        }

        .btn-classify {
            background: linear-gradient(135deg, #0284c7, #7c3aed);
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-classify:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        }

        .btn-classify:active {
            transform: translateY(0);
        }

        /* Result View */
        .placeholder-view {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 320px;
            color: var(--text-muted);
            text-align: center;
            border: 2px dashed var(--card-border);
            border-radius: 12px;
        }

        .result-view {
            display: none;
        }

        .pred-header {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(168, 85, 247, 0.1));
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .pred-class {
            font-size: 20px;
            font-weight: 700;
            color: var(--accent-blue);
            text-transform: capitalize;
        }

        .pred-meta {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            gap: 12px;
            margin-top: 4px;
        }

        .confidence-chip {
            background: var(--accent-green);
            color: #0f172a;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 700;
        }

        .probs-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .prob-row {
            margin-bottom: 10px;
        }

        .prob-labels {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 4px;
        }

        .prob-track {
            height: 8px;
            background-color: #0f172a;
            border-radius: 4px;
            overflow: hidden;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            width: 0%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .prob-bar.top-bar {
            background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        }

        /* Model Comparison Footer */
        .comparison-footer {
            max-width: 1200px;
            margin: 24px auto 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .model-chip {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            padding: 14px 18px;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .model-chip-info h4 {
            font-size: 14px;
            font-weight: 600;
        }

        .model-chip-info p {
            font-size: 12px;
            color: var(--text-muted);
        }

        .api-docs-link {
            text-align: center;
            margin-top: 24px;
        }

        .api-docs-link a {
            color: var(--accent-blue);
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
        }

        .api-docs-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>

    <div class="header">
        <div class="brand">
            <div class="brand-logo">📄</div>
            <div class="brand-title">
                <h1>Document OCR Classifier</h1>
                <p>Fine-tuned DistilBERT & TinyBERT REST Dashboard</p>
            </div>
        </div>
        <div class="badge-status">
            <span></span> System Online
        </div>
    </div>

    <div class="main-grid">
        <!-- Input Panel -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">📝 Input OCR Document Text</div>
            </div>

            <div class="templates-bar">
                <button class="template-btn" onclick="loadTemplate('invoice')">🧾 Invoice</button>
                <button class="template-btn" onclick="loadTemplate('receipt')">🛒 Receipt</button>
                <button class="template-btn" onclick="loadTemplate('resume')">💼 Resume</button>
                <button class="template-btn" onclick="loadTemplate('letter')">✉️ Letter</button>
                <button class="template-btn" onclick="loadTemplate('scientific')">🔬 Research Paper</button>
                <button class="template-btn" onclick="loadTemplate('contract')">⚖️ Legal Contract</button>
            </div>

            <textarea id="ocrInput" placeholder="Paste raw OCR text here or click one of the document templates above..."></textarea>

            <div class="controls-row">
                <div class="select-group">
                    <label for="modelSelect">Model:</label>
                    <select id="modelSelect">
                        <option value="distilbert">DistilBERT (67M params, High Accuracy)</option>
                        <option value="tinybert">TinyBERT (14M params, 15ms Fast)</option>
                    </select>
                </div>
                <button class="btn-classify" onclick="classifyText()">✨ Classify Document</button>
            </div>
        </div>

        <!-- Output Panel -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">🎯 Prediction Output</div>
            </div>

            <div id="placeholderView" class="placeholder-view">
                <div style="font-size: 36px; margin-bottom: 12px;">📊</div>
                <p style="font-size: 14px;">Select a document template and click <strong>Classify Document</strong> to view predictions and probability scores.</p>
            </div>

            <div id="resultView" class="result-view">
                <div class="pred-header">
                    <div>
                        <div class="pred-class" id="predClass">Invoice</div>
                        <div class="pred-meta">
                            <span>Model: <strong id="predModel">DistilBERT</strong></span>
                            <span>Latency: <strong id="predLatency">12 ms</strong></span>
                        </div>
                    </div>
                    <div class="confidence-chip" id="predConfidence">98.5%</div>
                </div>

                <div class="probs-title">Class Probabilities Distribution</div>
                <div id="probsContainer"></div>
            </div>
        </div>
    </div>

    <!-- Model Comparison Chips -->
    
    <div class="api-docs-link">
        <a href="/docs" target="_blank">🔗 Open Swagger Interactive API Documentation (/docs)</a>
    </div>

    <script>
        const TEMPLATES = {
            invoice: `INVOICE #INV-2026-8841\\nVendor: Apex Solutions Inc\\nDate: 2026-08-01\\nDue Date: 2026-09-01\\nBill To: Global Tech Ltd\\nDescription: Software Consulting & Server Maintenance - $4,500.00\\nTax: $382.50\\nTOTAL DUE: $4,882.50\\nPlease remit payment by due date.`,
            receipt: `*** RECEIPT ***\\nTarget Retail Store #481\\nDate: 2026-08-05 Time: 14:22\\nOrganic Milk 1Gal $4.29\\nCoffee Beans 1lb $14.99\\nUSB-C Cable $19.99\\nSUBTOTAL: $39.27\\nTAX: $3.14\\nTOTAL: $42.41\\nVISA ENDING IN *4921\\nTHANK YOU FOR SHOPPING WITH US!`,
            resume: `ALICE JOHNSON\\nEmail: alice.johnson@email.com | Phone: (555) 234-5678\\nLocation: San Francisco, CA\\n\\nOBJECTIVE\\nSenior Software Engineer with 8 years of experience building scalable backend microservices.\\n\\nWORK EXPERIENCE\\nNexus Logistics - Senior Software Engineer (2022 - Present)\\n- Led team of 8 building real-time data pipelines.\\n- Reduced API latency by 45% using PyTorch & Redis.\\n\\nEDUCATION\\nB.S. in Computer Science - UC Berkeley\\nSkills: Python, PyTorch, Docker, Kubernetes, AWS, SQL`,
            letter: `Starlight Media\\n100 Business Parkway, Suite 400\\nNew York, NY\\n\\nDate: August 8, 2026\\n\\nDear John Smith,\\n\\nI am writing to formally communicate our quarterly performance updates regarding our ongoing joint project. We have evaluated the operational metrics and are pleased to report significant progress across all deliverables.\\n\\nShould you have any questions, please do not hesitate to contact my office.\\n\\nSincerely,\\nDavid Lee\\nExecutive Vice President`,
            scientific: `RESEARCH REPORT / JOURNAL OF APPLIED AI\\nTitle: Evaluation of Transformer Architectures in OCR Text Classification\\nAuthors: Sarah Wilson, Michael Brown\\nAffiliation: Department of Computer Science, MIT\\n\\nABSTRACT\\nIn this study, we present a novel framework for evaluating deep learning model efficiency under real-world noise conditions. We conduct extensive experiments across diverse datasets, analyzing convergence rates, precision, recall, and F1-score performance.\\n\\n1. INTRODUCTION\\nRecent advances in neural architecture design have driven significant gains in natural language processing tasks.`,
            contract: `NON-DISCLOSURE AND SERVICE AGREEMENT\\nContract Ref: AGR-2026-491\\n\\nThis Agreement is entered into on this 8th day of August 2026, by and between Disclosing Party and Receiving Party.\\n\\n1. CONFIDENTIAL INFORMATION\\nThe Receiving Party agrees to hold in confidence all proprietary technical, financial, and business information disclosed. Confidential Information shall not be disclosed without prior written authorization.\\n\\n2. GOVERNING LAW\\nThis agreement shall be governed by the laws of California.`
        };

        function loadTemplate(type) {
            document.getElementById('ocrInput').value = TEMPLATES[type];
        }

        async function classifyText() {
            const text = document.getElementById('ocrInput').value.trim();
            if (!text) {
                alert('Please enter or select some OCR text first.');
                return;
            }

            const modelChoice = document.getElementById('modelSelect').value;
            const payload = { 
                text: text,
                model_name: modelChoice 
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const err = await response.json();
                    alert('Error: ' + (err.detail || 'Failed to classify'));
                    return;
                }

                const data = await response.json();
                renderResult(data);
            } catch (e) {
                alert('Network or server error: ' + e.message);
            }
        }

        function renderResult(data) {
            document.getElementById('placeholderView').style.display = 'none';
            document.getElementById('resultView').style.display = 'block';

            document.getElementById('predClass').innerText = data.predicted_document_type.replace('_', ' ');
            document.getElementById('predModel').innerText = data.model_used;
            document.getElementById('predLatency').innerText = data.inference_time_ms + ' ms';
            document.getElementById('predConfidence').innerText = (data.confidence_score * 100).toFixed(1) + '%';

            const container = document.getElementById('probsContainer');
            container.innerHTML = '';

            const sorted = Object.entries(data.class_probabilities).sort((a, b) => b[1] - a[1]);

            sorted.forEach(([cls, prob], index) => {
                const pct = (prob * 100).toFixed(1);
                const isTop = index === 0;

                const row = document.createElement('div');
                row.className = 'prob-row';
                row.innerHTML = `
                    <div class="prob-labels">
                        <span>${cls.replace('_', ' ')}</span>
                        <strong>${pct}%</strong>
                    </div>
                    <div class="prob-track">
                        <div class="prob-bar ${isTop ? 'top-bar' : ''}" style="width: ${pct}%"></div>
                    </div>
                `;
                container.appendChild(row);
            });
        }

        // Auto-load invoice sample on open
        loadTemplate('invoice');
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def render_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/info", tags=["Info"])
def api_info():
    return {
        "title": "Document Classification API",
        "status": "online",
        "active_default_model": LOADED_MODELS[active_model_key]["name"],
        "available_models": [v["name"] for v in LOADED_MODELS.values()],
        "supported_classes": list(label2id.keys()),
        "docs_url": "http://127.0.0.1:8000/docs",
        "predict_endpoint": "http://127.0.0.1:8000/predict"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "active_default_model": LOADED_MODELS[active_model_key]["name"],
        "loaded_models": list(LOADED_MODELS.keys()),
        "device": str(device)
    }

@app.post("/switch_model", tags=["Model Management"])
def switch_active_model(payload: ModelSwitchRequest):
    global active_model_key
    target = payload.model_name.lower().strip()
    if target not in LOADED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name '{payload.model_name}'. Choose from: {list(LOADED_MODELS.keys())}"
        )
    active_model_key = target
    return {
        "status": "success",
        "message": f"Active default model switched to '{LOADED_MODELS[active_model_key]['name']}'",
        "active_model": LOADED_MODELS[active_model_key]["name"]
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
    disp_name = model_entry["name"]
    model_id2label = model_entry.get("id2label", id2label)

    try:
        start_t = time.time()
        
        if USE_LOCAL_TORCH:
            model_obj = model_entry["model"]
            tokenizer_obj = model_entry["tokenizer"]
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
            predicted_class = model_id2label.get(pred_id, "unknown")
            confidence = float(probs[pred_id])
            class_probs = {model_id2label[i]: float(round(probs[i], 4)) for i in range(len(model_id2label))}
        else:
            import httpx
            headers = {}
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            hf_url = f"https://api-inference.huggingface.co/models/{model_entry['repo']}"
            resp = httpx.post(hf_url, json={"inputs": raw_text}, headers=headers, timeout=20.0)
            
            if resp.status_code != 200:
                raise RuntimeError(f"Hugging Face API returned status {resp.status_code}: {resp.text}")

            result = resp.json()
            if isinstance(result, list) and len(result) > 0:
                candidates = result[0] if isinstance(result[0], list) else result
                top_candidate = candidates[0]
                
                raw_lbl = top_candidate.get("label", "")
                if raw_lbl.startswith("LABEL_"):
                    lbl_id = int(raw_lbl.replace("LABEL_", ""))
                    predicted_class = model_id2label.get(lbl_id, raw_lbl)
                else:
                    predicted_class = raw_lbl

                confidence = float(top_candidate.get("score", 0.0))

                class_probs = {}
                for item in candidates:
                    lbl = item.get("label", "")
                    if lbl.startswith("LABEL_"):
                        l_id = int(lbl.replace("LABEL_", ""))
                        c_name = model_id2label.get(l_id, lbl)
                    else:
                        c_name = lbl
                    class_probs[c_name] = float(round(item.get("score", 0.0), 4))
            else:
                raise RuntimeError("Unexpected response structure from Hugging Face API.")

        latency_ms = round((time.time() - start_t) * 1000.0, 2)

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
