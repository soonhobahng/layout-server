from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ..models.schemas import ImageAnalysisResponse, PDFAnalysisResponse, HealthResponse, ErrorResponse, VisualizationResponse, LayoutElement, BoundingBox, Page
from ..utils.image_utils import ImageUtils
from ..utils.visualization_utils import LayoutVisualizer
from ..services.ocr_factory import OCRFactory
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/api/health", response_model=HealthResponse)
async def health_check(request: Request):
    try:
        layout_service = request.app.state.layout_service
        model_info = layout_service.get_model_info()
        
        # 모델 타입 확인
        is_mock = isinstance(layout_service.model, type(layout_service.model)) and \
                  layout_service.model.__class__.__name__ == 'MockLayoutModel'
        
        model_name = f"{model_info['model_name']} {'(Mock)' if is_mock else '(Real)'}"
        
        return HealthResponse(
            status="healthy",
            model=model_name,
            gpu_available=model_info["gpu_available"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@router.get("/api/model/status")
async def model_status(request: Request):
    try:
        layout_service = request.app.state.layout_service
        model_info = layout_service.get_model_info()
        
        # 모델 상태 확인
        is_mock = isinstance(layout_service.model, type(layout_service.model)) and \
                  layout_service.model.__class__.__name__ == 'MockLayoutModel'
        
        return {
            "model_name": model_info["model_name"],
            "model_type": "mock" if is_mock else "real",
            "confidence_threshold": model_info["confidence_threshold"],
            "gpu_available": model_info["gpu_available"],
            "status": "using_fallback_mock" if is_mock else "using_real_model",
            "recommendation": "Run quick_model_fix.py to download real model" if is_mock else "Model working properly"
        }
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@router.post("/api/analyze/image", response_model=ImageAnalysisResponse)
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(None, ge=0.1, le=0.99)
):
    try:
        if not ImageUtils.validate_image_file(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: jpg, png, bmp, tiff, webp"
            )
        
        image_bytes = await file.read()
        
        image = ImageUtils.load_image_from_bytes(image_bytes)
        image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
        width, height = ImageUtils.get_image_dimensions(image)
        
        layout_service = request.app.state.layout_service
        ocr_service = request.app.state.ocr_service
        
        elements_data = layout_service.detect(image, confidence_threshold)
        
        elements = []
        for element_data in elements_data:
            bbox_coords = element_data['coordinates']
            cropped_image = ImageUtils.crop_element(image, bbox_coords)
            
            content = None
            image_base64 = None
            
            element_type = element_data['type']
            
            if element_type in ['text', 'title', 'list']:
                content = ocr_service.extract_text(cropped_image, element_type)
            elif element_type == 'table':
                content = ocr_service.extract_text(cropped_image, element_type)
                image_base64 = ImageUtils.numpy_to_base64(cropped_image)
            elif element_type == 'figure':
                content = None
                image_base64 = ImageUtils.numpy_to_base64(cropped_image)
            
            bbox = BoundingBox(**element_data['bbox'])
            
            layout_element = LayoutElement(
                id=element_data['id'],
                type=element_type,
                bbox=bbox,
                confidence=element_data['confidence'],
                content=content,
                image_base64=image_base64
            )
            
            elements.append(layout_element)
        
        return ImageAnalysisResponse(
            success=True,
            width=width,
            height=height,
            elements=elements
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/api/analyze/pdf", response_model=PDFAnalysisResponse)
async def analyze_pdf(
    request: Request,
    file: UploadFile = File(...),
    dpi: Optional[int] = Query(200, ge=100, le=400),
    confidence_threshold: Optional[float] = Query(None, ge=0.1, le=0.99)
):
    try:
        if not ImageUtils.validate_pdf_file(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Expected: application/pdf"
            )
        
        pdf_bytes = await file.read()
        
        images = ImageUtils.pdf_to_images(pdf_bytes, dpi)
        
        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No pages found in PDF")
        
        layout_service = request.app.state.layout_service
        ocr_service = request.app.state.ocr_service
        
        pages = []
        for page_num, image in enumerate(images, 1):
            image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
            width, height = ImageUtils.get_image_dimensions(image)
            
            elements_data = layout_service.detect(image, confidence_threshold)
            
            elements = []
            for element_data in elements_data:
                bbox_coords = element_data['coordinates']
                cropped_image = ImageUtils.crop_element(image, bbox_coords)
                
                content = None
                image_base64 = None
                
                element_type = element_data['type']
                
                if element_type in ['text', 'title', 'list']:
                    content = ocr_service.extract_text(cropped_image, element_type)
                elif element_type == 'table':
                    content = ocr_service.extract_text(cropped_image, element_type)
                    image_base64 = ImageUtils.numpy_to_base64(cropped_image)
                elif element_type == 'figure':
                    content = None
                    image_base64 = ImageUtils.numpy_to_base64(cropped_image)
                
                bbox = BoundingBox(**element_data['bbox'])
                
                layout_element = LayoutElement(
                    id=element_data['id'],
                    type=element_type,
                    bbox=bbox,
                    confidence=element_data['confidence'],
                    content=content,
                    image_base64=image_base64
                )
                
                elements.append(layout_element)
            
            page = Page(
                page=page_num,
                width=width,
                height=height,
                elements=elements
            )
            pages.append(page)
        
        return PDFAnalysisResponse(
            success=True,
            total_pages=len(pages),
            pages=pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/api/visualize/image", response_model=VisualizationResponse)
async def visualize_image_layout(
    request: Request,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(None, ge=0.1, le=0.99),
    show_labels: Optional[bool] = Query(True),
    show_confidence: Optional[bool] = Query(True),
    box_thickness: Optional[int] = Query(2, ge=1, le=10)
):
    try:
        if not ImageUtils.validate_image_file(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: jpg, png, bmp, tiff, webp"
            )
        
        image_bytes = await file.read()
        
        image = ImageUtils.load_image_from_bytes(image_bytes)
        image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
        width, height = ImageUtils.get_image_dimensions(image)
        
        layout_service = request.app.state.layout_service
        ocr_service = request.app.state.ocr_service
        
        elements_data = layout_service.detect(image, confidence_threshold)
        
        elements = []
        for element_data in elements_data:
            bbox_coords = element_data['coordinates']
            cropped_image = ImageUtils.crop_element(image, bbox_coords)
            
            content = None
            element_type = element_data['type']
            
            if element_type in ['text', 'title', 'list', 'table']:
                content = ocr_service.extract_text(cropped_image, element_type)
            
            bbox = BoundingBox(**element_data['bbox'])
            
            layout_element = {
                'element_id': element_data['id'],
                'element_type': element_type,
                'bounding_box': {
                    'x1': bbox.x1,
                    'y1': bbox.y1,
                    'x2': bbox.x2,
                    'y2': bbox.y2
                },
                'confidence_score': element_data['confidence'],
                'text_content': content,
                'image_data': None
            }
            
            elements.append(layout_element)
        
        visualizer = LayoutVisualizer()
        visualized_image = visualizer.visualize_layout(
            image, 
            elements, 
            show_labels=show_labels,
            show_confidence=show_confidence,
            box_thickness=box_thickness
        )
        
        visualization_base64 = visualizer.to_base64(visualized_image)
        
        return VisualizationResponse(
            is_success=True,
            visualization_image=visualization_base64,
            original_width=width,
            original_height=height,
            total_elements=len(elements)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/api/visualize/pdf")
async def visualize_pdf_layout(
    request: Request,
    file: UploadFile = File(...),
    page_number: Optional[int] = Query(1, ge=1),
    dpi: Optional[int] = Query(200, ge=100, le=400),
    confidence_threshold: Optional[float] = Query(None, ge=0.1, le=0.99),
    show_labels: Optional[bool] = Query(True),
    show_confidence: Optional[bool] = Query(True),
    box_thickness: Optional[int] = Query(2, ge=1, le=10)
):
    try:
        if not ImageUtils.validate_pdf_file(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Expected: application/pdf"
            )
        
        pdf_bytes = await file.read()
        
        images = ImageUtils.pdf_to_images(pdf_bytes, dpi)
        
        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No pages found in PDF")
        
        if page_number > len(images):
            raise HTTPException(
                status_code=400, 
                detail=f"Page {page_number} not found. PDF has {len(images)} pages."
            )
        
        target_image = images[page_number - 1]
        target_image = ImageUtils.resize_image_if_needed(target_image, settings.max_image_size)
        width, height = ImageUtils.get_image_dimensions(target_image)
        
        layout_service = request.app.state.layout_service
        ocr_service = request.app.state.ocr_service
        
        elements_data = layout_service.detect(target_image, confidence_threshold)
        
        elements = []
        for element_data in elements_data:
            bbox_coords = element_data['coordinates']
            cropped_image = ImageUtils.crop_element(target_image, bbox_coords)
            
            content = None
            element_type = element_data['type']
            
            if element_type in ['text', 'title', 'list', 'table']:
                content = ocr_service.extract_text(cropped_image, element_type)
            
            bbox = BoundingBox(**element_data['bbox'])
            
            layout_element = {
                'element_id': element_data['id'],
                'element_type': element_type,
                'bounding_box': {
                    'x1': bbox.x1,
                    'y1': bbox.y1,
                    'x2': bbox.x2,
                    'y2': bbox.y2
                },
                'confidence_score': element_data['confidence'],
                'text_content': content,
                'image_data': None
            }
            
            elements.append(layout_element)
        
        visualizer = LayoutVisualizer()
        visualized_image = visualizer.visualize_layout(
            target_image, 
            elements, 
            show_labels=show_labels,
            show_confidence=show_confidence,
            box_thickness=box_thickness
        )
        
        visualization_base64 = visualizer.to_base64(visualized_image)
        
        return {
            "success": True,
            "visualization": visualization_base64,
            "page": page_number,
            "total_pages": len(images),
            "width": width,
            "height": height,
            "element_count": len(elements)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/api/debug/ocr")
async def debug_ocr_quality(
    request: Request,
    file: UploadFile = File(...),
    element_index: Optional[int] = Query(0, description="분석할 요소 인덱스"),
    save_debug_images: Optional[bool] = Query(False, description="디버그 이미지 저장 여부")
):
    """OCR 품질 디버깅을 위한 엔드포인트"""
    try:
        if not ImageUtils.validate_image_file(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        image_bytes = await file.read()
        image = ImageUtils.load_image_from_bytes(image_bytes)
        image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
        
        layout_service = request.app.state.layout_service
        ocr_service = request.app.state.ocr_service
        
        # 레이아웃 감지
        elements_data = layout_service.detect(image)
        
        if element_index >= len(elements_data):
            raise HTTPException(
                status_code=400, 
                detail=f"Element index {element_index} out of range. Found {len(elements_data)} elements."
            )
        
        if not elements_data:
            return {
                "success": False,
                "message": "No elements detected in image",
                "total_elements": 0
            }
        
        # 선택된 요소 분석
        element_data = elements_data[element_index]
        bbox_coords = element_data['coordinates']
        cropped_image = ImageUtils.crop_element(image, bbox_coords)
        element_type = element_data['type']
        
        # 여러 전처리 방식으로 OCR 테스트
        ocr_results = {}
        
        # 1. 기본 전처리
        preprocessed1 = ocr_service._preprocess_image(cropped_image, element_type)
        text1 = ocr_service._perform_ocr(preprocessed1, element_type)
        ocr_results['basic_preprocess'] = {
            "text": text1,
            "length": len(text1.strip()),
            "method": "Basic preprocessing with OTSU threshold"
        }
        
        # 2. 향상된 전처리
        preprocessed2 = ocr_service._advanced_preprocess(cropped_image, element_type)
        text2 = ocr_service._perform_ocr(preprocessed2, element_type)
        ocr_results['advanced_preprocess'] = {
            "text": text2,
            "length": len(text2.strip()),
            "method": "Enhanced preprocessing with contrast/sharpness"
        }
        
        # 3. 스케일링 우선
        preprocessed3 = ocr_service._scale_first_preprocess(cropped_image, element_type)
        text3 = ocr_service._perform_ocr(preprocessed3, element_type)
        ocr_results['scale_first'] = {
            "text": text3,
            "length": len(text3.strip()),
            "method": "Scale-first preprocessing with 4x upscaling"
        }
        
        # 4. Fallback 방식
        text4 = ocr_service._fallback_ocr(cropped_image, element_type)
        ocr_results['fallback'] = {
            "text": text4,
            "length": len(text4.strip()),
            "method": "Multiple PSM modes fallback"
        }
        
        # 5. 통합 결과 (실제 사용되는 방식)
        final_text = ocr_service.extract_text(cropped_image, element_type)
        
        # 디버그 이미지 Base64 인코딩
        debug_images = {}
        if save_debug_images:
            from ..utils.visualization_utils import LayoutVisualizer
            visualizer = LayoutVisualizer()
            
            debug_images['original'] = visualizer.to_base64(cropped_image)
            debug_images['basic_preprocessed'] = visualizer.to_base64(preprocessed1)
            debug_images['advanced_preprocessed'] = visualizer.to_base64(preprocessed2)
            debug_images['scale_first_preprocessed'] = visualizer.to_base64(preprocessed3)
        
        return {
            "success": True,
            "element_info": {
                "index": element_index,
                "type": element_type,
                "bbox": element_data['bbox'],
                "confidence": element_data['confidence'],
                "image_size": f"{cropped_image.shape[1]}x{cropped_image.shape[0]}"
            },
            "ocr_results": ocr_results,
            "final_result": {
                "text": final_text,
                "length": len(final_text.strip()),
                "method": "Multi-preprocessing with best result selection"
            },
            "debug_images": debug_images if save_debug_images else {},
            "recommendations": {
                "best_method": max(ocr_results.keys(), key=lambda k: ocr_results[k]['length']),
                "text_quality": "good" if len(final_text.strip()) > 5 else "poor",
                "suggestions": [
                    "이미지 해상도 높이기" if cropped_image.shape[0] < 50 else "해상도 양호",
                    "대비 개선 필요" if len(text2) > len(text1) else "대비 적절",
                    "스케일링 효과적" if len(text3) > max(len(text1), len(text2)) else "스케일링 불필요"
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR debug failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@router.get("/api/ocr/engines")
async def get_ocr_engines():
    """사용 가능한 OCR 엔진 정보 조회"""
    try:
        engine_info = OCRFactory.get_engine_info()
        return {
            "success": True,
            "current_engine": engine_info["current_engine"],
            "available_engines": engine_info["available_engines"],
            "engine_configs": {
                "tesseract": engine_info["tesseract_config"],
                "paddleocr": engine_info["paddleocr_config"]
            },
            "recommendations": {
                "for_korean": "paddleocr" if "paddleocr" in engine_info["available_engines"] else "tesseract",
                "for_english": "tesseract",
                "for_mixed": "paddleocr" if "paddleocr" in engine_info["available_engines"] else "tesseract"
            }
        }
    except Exception as e:
        logger.error(f"OCR engines info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engine info: {str(e)}")

@router.post("/api/ocr/compare")
async def compare_ocr_engines(
    request: Request,
    file: UploadFile = File(...),
    element_index: Optional[int] = Query(0, description="비교할 요소 인덱스")
):
    """Tesseract와 PaddleOCR 성능 비교"""
    try:
        if not ImageUtils.validate_image_file(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        image_bytes = await file.read()
        image = ImageUtils.load_image_from_bytes(image_bytes)
        image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
        
        layout_service = request.app.state.layout_service
        
        # 레이아웃 감지
        elements_data = layout_service.detect(image)
        
        if element_index >= len(elements_data):
            raise HTTPException(
                status_code=400, 
                detail=f"Element index {element_index} out of range. Found {len(elements_data)} elements."
            )
        
        if not elements_data:
            return {
                "success": False,
                "message": "No elements detected in image",
                "total_elements": 0
            }
        
        # 선택된 요소 추출
        element_data = elements_data[element_index]
        bbox_coords = element_data['coordinates']
        cropped_image = ImageUtils.crop_element(image, bbox_coords)
        element_type = element_data['type']
        
        # 두 OCR 엔진으로 비교
        try:
            # PaddleOCR 서비스 생성
            from ..services.paddle_ocr_service import PaddleOCRService
            paddle_service = PaddleOCRService(lang=settings.paddle_ocr_lang)
            
            # Tesseract 서비스 생성 
            from ..services.ocr_service import OCRService
            tesseract_service = OCRService(lang=settings.ocr_lang)
            
            # 성능 비교
            comparison = paddle_service.compare_with_tesseract(cropped_image, tesseract_service)
            
            return {
                "success": True,
                "element_info": {
                    "index": element_index,
                    "type": element_type,
                    "bbox": element_data['bbox'],
                    "confidence": element_data['confidence']
                },
                "comparison_results": comparison,
                "recommendation": {
                    "better_engine": "paddleocr" if comparison.get("paddle_ocr", {}).get("length", 0) > comparison.get("tesseract", {}).get("length", 0) else "tesseract",
                    "reason": "PaddleOCR extracted more text" if comparison.get("paddle_ocr", {}).get("length", 0) > comparison.get("tesseract", {}).get("length", 0) else "Tesseract extracted more text"
                }
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "PaddleOCR not installed. Install with: pip install paddleocr",
                "available_engines": OCRFactory.get_available_engines()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")