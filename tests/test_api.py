import os
import sys
# Add root path to Python PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    assert "active_model" in data
    assert "device" in data

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "model_type" in data
    assert "onnx_optimized" in data

def test_predict_invalid_file_type():
    # Attempt to upload an invalid file extension (text file)
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_predict_invalid_mime_type():
    # Attempt to upload an image with a non-image MIME type (spoofed extension)
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid MIME type" in response.json()["detail"]
