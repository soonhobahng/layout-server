from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ..models.schemas import ImageAnalysisResponse, PDFAnalysisResponse, HealthResponse, ErrorResponse, LayoutElement, BoundingBox, Page
from ..utils.image_utils import ImageUtils
from ..services.ocr_service import OCRService
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/api/health", response_model=HealthResponse)
async def health_check(request: Request):
    try:
        layout_service = request.app.state.layout_service
        model_info = layout_service.get_model_info()
        
        return HealthResponse(
            status="healthy",
            model=model_info["model_name"],
            gpu_available=model_info["gpu_available"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

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