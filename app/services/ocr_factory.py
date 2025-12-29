from typing import Union
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class OCRFactory:
    """OCR 엔진 팩토리 클래스"""
    
    @staticmethod
    def create_ocr_service() -> Union['OCRService', 'PaddleOCRService']:
        """설정에 따라 적절한 OCR 서비스 생성"""
        
        if settings.ocr_engine.lower() == "paddleocr":
            try:
                from .paddle_ocr_service import PaddleOCRService
                logger.info("Creating PaddleOCR service")
                return PaddleOCRService(lang=settings.paddle_ocr_lang)
            except ImportError:
                logger.warning("PaddleOCR not available, falling back to Tesseract")
                from .ocr_service import OCRService
                return OCRService(lang=settings.ocr_lang)
            except Exception as e:
                logger.error(f"Failed to create PaddleOCR service: {e}")
                logger.info("Falling back to Tesseract")
                from .ocr_service import OCRService
                return OCRService(lang=settings.ocr_lang)
        
        elif settings.ocr_engine.lower() == "tesseract":
            logger.info("Creating Tesseract OCR service")
            from .ocr_service import OCRService
            return OCRService(lang=settings.ocr_lang)
        
        else:
            logger.warning(f"Unknown OCR engine: {settings.ocr_engine}, using Tesseract")
            from .ocr_service import OCRService
            return OCRService(lang=settings.ocr_lang)
    
    @staticmethod
    def get_available_engines() -> list:
        """사용 가능한 OCR 엔진 목록 반환"""
        available = ["tesseract"]
        
        try:
            import paddleocr
            available.append("paddleocr")
        except ImportError:
            pass
        
        return available
    
    @staticmethod
    def get_engine_info() -> dict:
        """현재 설정된 엔진 정보 반환"""
        available_engines = OCRFactory.get_available_engines()
        
        return {
            "current_engine": settings.ocr_engine,
            "available_engines": available_engines,
            "is_current_available": settings.ocr_engine in available_engines,
            "tesseract_config": {
                "language": settings.ocr_lang
            },
            "paddleocr_config": {
                "language": settings.paddle_ocr_lang,
                "gpu_enabled": settings.use_gpu_ocr
            }
        }