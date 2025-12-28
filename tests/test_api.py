import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app

client = TestClient(app)

@pytest.fixture
def sample_image():
    image = Image.new('RGB', (800, 600), color='white')
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    return img_byte_array

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "gpu_available" in data

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Layout Analysis Server" in data["message"]
    assert data["version"] == "1.0.0"

def test_analyze_image_success(sample_image):
    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/api/analyze/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "width" in data
    assert "height" in data
    assert "elements" in data

def test_analyze_image_with_confidence_threshold(sample_image):
    files = {"file": ("test.png", sample_image, "image/png")}
    data = {"confidence_threshold": 0.8}
    response = client.post("/api/analyze/image", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True

def test_analyze_image_invalid_file_type():
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/analyze/image", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_analyze_image_invalid_confidence():
    sample_image = io.BytesIO()
    Image.new('RGB', (100, 100)).save(sample_image, format='PNG')
    sample_image.seek(0)
    
    files = {"file": ("test.png", sample_image, "image/png")}
    data = {"confidence_threshold": 1.5}  # Invalid: > 0.99
    response = client.post("/api/analyze/image", files=files, data=data)
    assert response.status_code == 422  # Validation error

def create_simple_pdf():
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(100, 750, "Test PDF Document")
        p.drawString(100, 700, "This is a test page.")
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer
    except ImportError:
        return None

def test_analyze_pdf_success():
    pdf_buffer = create_simple_pdf()
    if pdf_buffer is None:
        pytest.skip("reportlab not available for PDF test")
    
    files = {"file": ("test.pdf", pdf_buffer, "application/pdf")}
    response = client.post("/api/analyze/pdf", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "total_pages" in data
    assert "pages" in data
    assert data["total_pages"] == len(data["pages"])

def test_analyze_pdf_with_dpi():
    pdf_buffer = create_simple_pdf()
    if pdf_buffer is None:
        pytest.skip("reportlab not available for PDF test")
    
    files = {"file": ("test.pdf", pdf_buffer, "application/pdf")}
    data = {"dpi": 150}
    response = client.post("/api/analyze/pdf", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True

def test_analyze_pdf_invalid_file_type():
    files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
    response = client.post("/api/analyze/pdf", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main([__file__])