import sys
from fastapi.testclient import TestClient
from api import app

def run_tests():
    print("==================================================")
    print("       RUNNING FASTAPI ENDPOINT UNIT TESTS")
    print("==================================================")

    with TestClient(app) as client:
        # Test 1: Dashboard HTML Endpoint
        print("\n[Test 1] GET / (Dashboard UI)")
        res = client.get("/")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "text/html" in res.headers["content-type"]
        assert "Document OCR Classifier" in res.text

        # Test 1b: Info JSON Endpoint
        print("\n[Test 1b] GET /info (API Info)")
        res_info = client.get("/info")
        assert res_info.status_code == 200
        assert res_info.json()["status"] == "online"

        # Test 2: Health Endpoint
        print("\n[Test 2] GET /health (Health Check)")
        res = client.get("/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("Response:", data)
        assert data["status"] == "healthy"

        # Test 3: Prediction with Default Model
        print("\n[Test 3] POST /predict (Default Model)")
        sample_invoice = {
            "text": "INVOICE #99482\nVendor: Apex Logistics\nDate: 2026-08-01\nTotal Amount Due: $4,500.00\nPlease pay by wire transfer."
        }
        res = client.post("/predict", json=sample_invoice)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        print("Prediction Result:", data)
        assert data["predicted_document_type"] == "invoice"

        # Test 4: Prediction explicitly asking for TinyBERT
        print("\n[Test 4] POST /predict (Explicitly Requesting TinyBERT)")
        sample_tiny = {
            "text": "INVOICE #99482\nVendor: Apex Logistics\nDate: 2026-08-01\nTotal Amount Due: $4,500.00",
            "model_name": "tinybert"
        }
        res = client.post("/predict", json=sample_tiny)
        assert res.status_code == 200
        data = res.json()
        print("Prediction Result:", data)
        assert data["model_used"] == "TinyBERT"

        # Test 5: Prediction explicitly asking for DistilBERT
        print("\n[Test 5] POST /predict (Explicitly Requesting DistilBERT)")
        sample_distil = {
            "text": "INVOICE #99482\nVendor: Apex Logistics\nDate: 2026-08-01\nTotal Amount Due: $4,500.00",
            "model_name": "distilbert"
        }
        res = client.post("/predict", json=sample_distil)
        assert res.status_code == 200
        data = res.json()
        print("Prediction Result:", data)
        assert data["model_used"] == "DistilBERT"


        # Test 7: Error Handling - Empty Text
        print("\n[Test 7] POST /predict (Error Handling - Empty Text)")
        empty_payload = {"text": "   "}
        res = client.post("/predict", json=empty_payload)
        assert res.status_code == 400

        # Test 8: Error Handling - Invalid Model Name
        print("\n[Test 8] POST /predict (Error Handling - Invalid Model Name)")
        invalid_model_payload = {"text": "INVOICE #100", "model_name": "unknown_model"}
        res = client.post("/predict", json=invalid_model_payload)
        assert res.status_code == 400
        print("Correctly caught invalid model response:", res.json())

    print("\n==================================================")
    print("       ALL API TESTS PASSED SUCCESSFULLY!          ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
