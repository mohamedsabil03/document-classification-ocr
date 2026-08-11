import pytest
from fastapi.testclient import TestClient
from api import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_dashboard_ui(client):
    """Test GET / (Dashboard UI HTML endpoint)"""
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "text/html" in res.headers["content-type"]
    assert "Document OCR Classifier" in res.text

def test_api_info(client):
    """Test GET /info (API Info endpoint)"""
    res_info = client.get("/info")
    assert res_info.status_code == 200
    assert res_info.json()["status"] == "online"

def test_health_check(client):
    """Test GET /health (Health Check endpoint)"""
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "healthy"

def test_predict_default(client):
    """Test POST /predict with default model"""
    sample_invoice = {
        "text": "INVOICE #99482\nVendor: Apex Logistics\nDate: 2026-08-01\nTotal Amount Due: $4,500.00\nPlease pay by wire transfer."
    }
    res = client.post("/predict", json=sample_invoice)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["predicted_document_type"] == "invoice"

def test_predict_tinybert(client):
    """Test POST /predict with explicit TinyBERT model"""
    sample_tiny = {
        "text": "INVOICE #99482\nVendor: Apex Logistics\nDate: 2026-08-01\nTotal Amount Due: $4,500.00",
        "model_name": "tinybert"
    }
    res = client.post("/predict", json=sample_tiny)
    assert res.status_code == 200
    data = res.json()
    assert data["model_used"] == "TinyBERT"

def test_predict_distilbert(client):
    """Test POST /predict with explicit DistilBERT model"""
    sample_distil = {
        "text": "INVOICE #99482\nVendor: Apex Logistics\nDate: 2026-08-01\nTotal Amount Due: $4,500.00",
        "model_name": "distilbert"
    }
    res = client.post("/predict", json=sample_distil)
    assert res.status_code == 200
    data = res.json()
    assert data["model_used"] == "DistilBERT"

def test_sample_endpoints(client):
    """Test GET /sample/invoice and GET /sample/random"""
    res_sample = client.get("/sample/invoice")
    assert res_sample.status_code == 200
    sample_data = res_sample.json()
    assert sample_data["label"] == "invoice"
    assert len(sample_data["text"]) > 0

    res_random = client.get("/sample/random")
    assert res_random.status_code == 200
    assert len(res_random.json()["text"]) > 0

def test_predict_empty_text(client):
    """Test POST /predict error handling with empty text"""
    empty_payload = {"text": "   "}
    res = client.post("/predict", json=empty_payload)
    assert res.status_code == 400

def test_predict_invalid_model(client):
    """Test POST /predict error handling with invalid model name"""
    invalid_model_payload = {"text": "INVOICE #100", "model_name": "unknown_model"}
    res = client.post("/predict", json=invalid_model_payload)
    assert res.status_code == 400

if __name__ == "__main__":
    pytest.main(["-v", __file__])

