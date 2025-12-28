# Layout Analysis Server

FastAPI 기반 PDF/이미지 레이아웃 분석 서버입니다. 문서 이미지에서 레이아웃(텍스트, 테이블, 이미지 영역)을 감지하고, 각 영역에서 텍스트를 추출하여 RAG용 데이터로 반환합니다.

## 기술 스택

- **FastAPI + Uvicorn**: 고성능 웹 프레임워크
- **layoutparser**: Detectron2 기반 레이아웃 감지
- **pytesseract**: OCR (한글+영어 지원)
- **pdf2image**: PDF → 이미지 변환
- **Docker**: 컨테이너화된 배포

## 주요 기능

- 🖼️ 이미지 파일 레이아웃 분석 (JPG, PNG, BMP, TIFF, WebP)
- 📄 PDF 파일 다중 페이지 분석
- 🔍 텍스트, 제목, 리스트, 테이블, 그림 영역 감지
- 📝 OCR을 통한 텍스트 추출 (한글+영어)
- 🖼️ 테이블/그림 영역 이미지 Base64 인코딩
- ⚙️ 신뢰도 임계값 조정 가능
- 🚀 Docker로 쉬운 배포

## 설치 및 실행

### 1. 로컬 설치

#### 필수 요구사항
- Python 3.10+
- Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-kor poppler-utils
```

**macOS:**
```bash
brew install tesseract tesseract-lang poppler
```

**Windows:**
- [Tesseract 설치](https://github.com/tesseract-ocr/tesseract)
- [Poppler 설치](https://github.com/oschwartz10612/poppler-windows)

#### 설치
```bash
cd layout-server
pip install -r requirements.txt

# 환경 변수 설정
cp .env .env

# 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Docker 실행 (권장)

```bash
cd layout-server

# Docker Compose로 실행
docker-compose up -d

# 또는 Docker 직접 실행
docker build -t layout-server .
docker run -p 8000:8000 layout-server
```

### 3. 서버 상태 확인

```bash
curl http://localhost:8000/api/health
```

응답:
```json
{
  "gpu_availability": false,
  "model_name": "faster_rcnn_R_50_FPN_3x",
  "service_status": "healthy"
}
```

## API 사용법

### 이미지 분석 API

```bash
curl -X POST "http://localhost:8000/api/analyze/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.png" \
  -F "confidence_threshold=0.7"
```

### PDF 분석 API

```bash
curl -X POST "http://localhost:8000/api/analyze/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "dpi=200" \
  -F "confidence_threshold=0.7"
```

### Python 클라이언트 예시

```python
import requests
import json

# 이미지 분석
with open('sample.png', 'rb') as f:
    files = {'file': f}
    data = {'confidence_threshold': 0.7}
    response = requests.post(
        'http://localhost:8000/api/analyze/image', 
        files=files, 
        data=data
    )

result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))

# PDF 분석
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {'dpi': 200, 'confidence_threshold': 0.7}
    response = requests.post(
        'http://localhost:8000/api/analyze/pdf', 
        files=files, 
        data=data
    )

result = response.json()
print(f"총 {result['total_page_count']}페이지 분석 완료")
```

### 응답 형식

> 📝 **JSON 출력 개선**: 가독성 향상을 위해 들여쓰기, 한글 지원, 의미있는 필드명을 적용했습니다.

#### 이미지 분석 응답
```json
{
  "image_height": 1400,
  "image_width": 1000,
  "is_success": true,
  "layout_elements": [
    {
      "bounding_box": {
        "x1": 100,
        "x2": 900,
        "y1": 50,
        "y2": 120
      },
      "confidence_score": 0.95,
      "element_id": 1,
      "element_type": "title",
      "image_data": null,
      "text_content": "문서 제목"
    },
    {
      "bounding_box": {
        "x1": 100,
        "x2": 900,
        "y1": 150,
        "y2": 300
      },
      "confidence_score": 0.88,
      "element_id": 2,
      "element_type": "text",
      "image_data": null,
      "text_content": "본문 텍스트 내용..."
    },
    {
      "bounding_box": {
        "x1": 100,
        "x2": 900,
        "y1": 350,
        "y2": 600
      },
      "confidence_score": 0.92,
      "element_id": 3,
      "element_type": "table",
      "image_data": "iVBORw0KGgo...",
      "text_content": "테이블 텍스트 내용"
    },
    {
      "bounding_box": {
        "x1": 100,
        "x2": 500,
        "y1": 650,
        "y2": 900
      },
      "confidence_score": 0.87,
      "element_id": 4,
      "element_type": "figure",
      "image_data": "iVBORw0KGgo...",
      "text_content": null
    }
  ]
}
```

#### PDF 분석 응답
```json
{
  "document_pages": [
    {
      "page_elements": [...],
      "page_height": 1400,
      "page_number": 1,
      "page_width": 1000
    },
    {
      "page_elements": [...],
      "page_height": 1400,
      "page_number": 2,
      "page_width": 1000
    }
  ],
  "is_success": true,
  "total_page_count": 3
}
```

#### JSON 응답 필드 설명

| 구 필드명 | 신 필드명 | 설명 |
|----------|-----------|------|
| `success` | `is_success` | 분석 성공 여부 |
| `width` | `image_width` / `page_width` | 이미지/페이지 너비 |
| `height` | `image_height` / `page_height` | 이미지/페이지 높이 |
| `elements` | `layout_elements` / `page_elements` | 레이아웃 요소 목록 |
| `id` | `element_id` | 고유 요소 식별자 |
| `type` | `element_type` | 요소 유형 |
| `bbox` | `bounding_box` | 경계 박스 좌표 |
| `confidence` | `confidence_score` | 검출 신뢰도 점수 |
| `content` | `text_content` | 추출된 텍스트 내용 |
| `image_base64` | `image_data` | Base64 인코딩된 이미지 |
| `total_pages` | `total_page_count` | 전체 페이지 수 |
| `pages` | `document_pages` | 문서 페이지 목록 |
| `page` | `page_number` | 페이지 번호 |

## 환경 변수 설정

`.env` 파일에서 다음 설정을 변경할 수 있습니다:

```bash
# 모델 설정 (개발용/프로덕션용)
MODEL_NAME=faster_rcnn_R_50_FPN_3x  # 또는 mask_rcnn_X_101_32x8d_FPN_3x

# 신뢰도 임계값 (0.1 ~ 0.99)
CONFIDENCE_THRESHOLD=0.7

# 최대 이미지 크기 (픽셀)
MAX_IMAGE_SIZE=4096

# OCR 언어 설정
OCR_LANG=kor+eng
```

### 모델 옵션

| 모델 | 크기 | 속도 | 정확도 | 용도 |
|------|------|------|--------|------|
| faster_rcnn_R_50_FPN_3x | ~170MB | 빠름 | 보통 | 개발/테스트 |
| mask_rcnn_X_101_32x8d_FPN_3x | ~856MB | 느림 | 높음 | 프로덕션 |

## 지원 파일 형식

### 이미지
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- WebP (.webp)

### PDF
- PDF (.pdf) - 다중 페이지 지원

## 감지되는 레이아웃 요소

1. **Text**: 일반 본문 텍스트
2. **Title**: 제목, 헤딩
3. **List**: 목록, 번호 매기기
4. **Table**: 표, 차트 (텍스트 + 이미지)
5. **Figure**: 그림, 이미지 (이미지만)

## 개발 가이드

### 프로젝트 구조
```
layout-server/
├── app/
│   ├── main.py              # FastAPI 앱
│   ├── config.py            # 설정 관리
│   ├── api/routes.py        # API 엔드포인트
│   ├── services/
│   │   ├── layout_service.py    # 레이아웃 감지
│   │   └── ocr_service.py       # OCR 처리
│   ├── models/schemas.py    # 데이터 모델
│   └── utils/image_utils.py # 이미지 처리 유틸
├── tests/                   # 테스트 코드
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### 테스트 실행

```bash
# API 문서 확인
open http://localhost:8000/docs

# 헬스체크
curl http://localhost:8000/api/health

# 샘플 이미지로 테스트
curl -X POST "http://localhost:8000/api/analyze/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/fixtures/sample.png"
```

## 성능 최적화

### GPU 가속화
CUDA 지원 GPU가 있는 경우:

```bash
# GPU 버전 PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Detectron2 GPU 버전 설치
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 메모리 최적화
- 대용량 이미지 자동 리사이징 (`MAX_IMAGE_SIZE`)
- 모델 캐시 디렉토리 볼륨 마운트 (Docker)
- 배치 처리를 위한 비동기 API

## 문제 해결

### 자주 발생하는 오류

1. **Tesseract 인식 오류**
   ```bash
   # Tesseract 설치 확인
   tesseract --version
   
   # 한글 언어팩 확인
   tesseract --list-langs
   ```

2. **모델 다운로드 오류**
   ```bash
   # 수동 모델 다운로드
   python -c "import layoutparser as lp; lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')"
   ```

3. **메모리 부족**
   - `MAX_IMAGE_SIZE` 값을 줄여보세요 (예: 2048)
   - 더 작은 모델을 사용하세요 (`faster_rcnn_R_50_FPN_3x`)

### 로그 확인
```bash
# Docker 로그
docker-compose logs -f layout-server

# 로컬 실행 시 상세 로그
uvicorn app.main:app --log-level debug
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여하기

1. 이슈 리포트는 GitHub Issues를 사용해주세요
2. Pull Request 환영합니다
3. 코드 스타일: Black, isort 사용