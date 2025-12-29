from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class BoundingBox(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x1": 100,
                "y1": 50,
                "x2": 300,
                "y2": 150
            }
        }
    )
    
    x1: int
    y1: int
    x2: int
    y2: int

class LayoutElement(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "element_id": 1,
                "element_type": "text",
                "bounding_box": {
                    "x1": 100,
                    "y1": 50,
                    "x2": 300,
                    "y2": 150
                },
                "confidence_score": 0.95,
                "text_content": "샘플 텍스트",
                "image_data": None
            }
        }
    )
    
    element_id: int = Field(..., alias="id", description="고유 요소 식별자")
    element_type: str = Field(..., alias="type", description="요소 유형 (text, title, list, table, figure)")
    bounding_box: BoundingBox = Field(..., alias="bbox", description="요소의 경계 박스 좌표")
    confidence_score: float = Field(..., alias="confidence", description="검출 신뢰도 점수")
    text_content: Optional[str] = Field(None, alias="content", description="추출된 텍스트 내용")
    image_data: Optional[str] = Field(None, alias="image_base64", description="Base64 인코딩된 이미지 데이터")

class ImageAnalysisResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "is_success": True,
                "image_width": 1200,
                "image_height": 800,
                "layout_elements": []
            }
        }
    )
    
    is_success: bool = Field(..., alias="success", description="분석 성공 여부")
    image_width: int = Field(..., alias="width", description="이미지 너비")
    image_height: int = Field(..., alias="height", description="이미지 높이")
    layout_elements: List[LayoutElement] = Field(..., alias="elements", description="검출된 레이아웃 요소들")

class Page(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    page_number: int = Field(..., alias="page", description="페이지 번호")
    page_width: int = Field(..., alias="width", description="페이지 너비")
    page_height: int = Field(..., alias="height", description="페이지 높이")
    page_elements: List[LayoutElement] = Field(..., alias="elements", description="페이지 내 레이아웃 요소들")

class PDFAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    is_success: bool = Field(..., alias="success", description="분석 성공 여부")
    total_page_count: int = Field(..., alias="total_pages", description="전체 페이지 수")
    document_pages: List[Page] = Field(..., alias="pages", description="분석된 페이지들")

class HealthResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=()
    )
    
    service_status: str = Field(..., alias="status", description="서비스 상태")
    model_name: str = Field(..., alias="model", description="사용 중인 모델명")
    gpu_availability: bool = Field(..., alias="gpu_available", description="GPU 사용 가능 여부")

class ErrorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    is_success: bool = Field(False, alias="success", description="처리 성공 여부")
    error_message: str = Field(..., alias="error", description="오류 메시지")
    error_detail: Optional[str] = Field(None, alias="detail", description="상세 오류 정보")

class VisualizationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    is_success: bool = Field(..., alias="success", description="시각화 성공 여부")
    visualization_image: str = Field(..., alias="visualization", description="Base64 인코딩된 시각화 이미지")
    original_width: int = Field(..., alias="width", description="원본 이미지 너비")
    original_height: int = Field(..., alias="height", description="원본 이미지 높이")
    total_elements: int = Field(..., alias="element_count", description="검출된 요소 총 개수")