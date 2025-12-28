Sample Test Files

To test the layout analysis server, you can use:

1. Any image file (PNG, JPG, etc.) with text, tables, or figures
2. Any PDF file with multiple pages

Sample curl commands:

# Test with image
curl -X POST "http://localhost:8000/api/analyze/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.png" \
  -F "confidence_threshold=0.7"

# Test with PDF  
curl -X POST "http://localhost:8000/api/analyze/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf" \
  -F "dpi=200"

# Health check
curl http://localhost:8000/api/health

Place your test files in this directory for testing.
