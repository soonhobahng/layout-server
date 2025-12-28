import pytesseract
import numpy as np
from typing import Optional
import logging
import cv2

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, lang: str = "kor+eng"):
        self.lang = lang
        
    def extract_text(self, image: np.ndarray, element_type: str = "text") -> Optional[str]:
        try:
            if element_type == "figure":
                return None
            
            if image.shape[0] < 10 or image.shape[1] < 10:
                logger.warning("Image too small for OCR")
                return ""
            
            image_preprocessed = self._preprocess_image(image, element_type)
            
            config = self._get_tesseract_config(element_type)
            
            text = pytesseract.image_to_string(
                image_preprocessed, 
                lang=self.lang, 
                config=config
            ).strip()
            
            return text if text else ""
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
    
    def _preprocess_image(self, image: np.ndarray, element_type: str) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            if element_type == "table":
                kernel = np.ones((1, 1), np.uint8)
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            elif element_type in ["title", "text"]:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            height, width = binary.shape
            if height < 50 or width < 50:
                scale = max(50 / height, 50 / width, 2.0)
                new_height = int(height * scale)
                new_width = int(width * scale)
                binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return binary
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def _get_tesseract_config(self, element_type: str) -> str:
        base_config = '--oem 3 --psm'
        
        if element_type == "title":
            return f"{base_config} 8"
        elif element_type == "text":
            return f"{base_config} 6" 
        elif element_type == "list":
            return f"{base_config} 6"
        elif element_type == "table":
            return f"{base_config} 6"
        else:
            return f"{base_config} 6"
    
    def set_language(self, lang: str):
        self.lang = lang