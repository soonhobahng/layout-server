import cv2
import numpy as np
import base64
from PIL import Image
import io
from pdf2image import convert_from_bytes
from typing import List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    @staticmethod
    def resize_image_if_needed(image: np.ndarray, max_size: int = 4096) -> np.ndarray:
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim <= max_size:
            return image
        
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        return resized
    
    @staticmethod
    def crop_element(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            logger.warning(f"Empty crop for bbox {bbox}")
            return np.zeros((10, 10, 3), dtype=np.uint8)
        
        return cropped
    
    @staticmethod
    def numpy_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
        try:
            if len(image.shape) == 3:
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                image_pil = Image.fromarray(image.astype(np.uint8), mode='L')
            
            buffer = io.BytesIO()
            image_pil.save(buffer, format=format)
            buffer.seek(0)
            
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            return ""
    
    @staticmethod
    def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[np.ndarray]:
        try:
            pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)
            numpy_images = []
            
            for pil_img in pil_images:
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                numpy_img = np.array(pil_img)
                numpy_images.append(numpy_img)
            
            logger.info(f"Converted PDF to {len(numpy_images)} images")
            return numpy_images
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    @staticmethod
    def validate_image_file(content_type: str) -> bool:
        allowed_types = [
            'image/jpeg', 
            'image/jpg', 
            'image/png', 
            'image/bmp', 
            'image/tiff',
            'image/webp'
        ]
        return content_type.lower() in allowed_types
    
    @staticmethod
    def validate_pdf_file(content_type: str) -> bool:
        return content_type.lower() == 'application/pdf'
    
    @staticmethod
    def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape[:2]
        return width, height