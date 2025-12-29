import numpy as np
from typing import Optional, List, Tuple
import logging
import cv2
from PIL import Image
import re

logger = logging.getLogger(__name__)

class PaddleOCRService:
    def __init__(self, lang: str = "korean"):
        self.lang = lang
        self.ocr_engine = None
        self._initialized = False
        
    def _initialize_paddle_ocr(self):
        """PaddleOCR 엔진 초기화"""
        try:
            from paddleocr import PaddleOCR
            
            # 한국어 + 영어 지원
            if self.lang == "korean":
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True, 
                    lang='korean'
                )
            else:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True, 
                    lang='en'
                )
            
            logger.info(f"PaddleOCR initialized with language: {self.lang}")
            
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def extract_text(self, image: np.ndarray, element_type: str = "text") -> Optional[str]:
        """PaddleOCR을 사용한 텍스트 추출"""
        try:
            # Lazy initialization
            if not self._initialized:
                logger.info("Lazy loading PaddleOCR on first use...")
                self._initialize_paddle_ocr()
                self._initialized = True
                
            if element_type == "figure":
                return None
            
            if image.shape[0] < 10 or image.shape[1] < 10:
                logger.warning("Image too small for OCR")
                return ""
            
            # 이미지 전처리
            preprocessed_image = self._preprocess_image(image, element_type)
            
            # PaddleOCR 실행
            results = self.ocr_engine.ocr(preprocessed_image)
            
            # 결과 처리
            extracted_text = self._process_paddle_results(results, element_type)
            
            # 텍스트 후처리
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            return cleaned_text if cleaned_text else ""
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return ""
    
    def _preprocess_image(self, image: np.ndarray, element_type: str) -> np.ndarray:
        """PaddleOCR에 최적화된 이미지 전처리"""
        try:
            # BGR to RGB 변환 (PaddleOCR은 RGB 선호)
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # OpenCV는 BGR이므로 RGB로 변환
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
            else:
                # 그레이스케일을 RGB로 변환
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 이미지 크기 최적화
            height, width = rgb_image.shape[:2]
            
            # 너무 작은 이미지는 확대
            if height < 32 or width < 32:
                scale_factor = max(32 / height, 32 / width, 2.0)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 너무 큰 이미지는 축소 (PaddleOCR 성능 최적화)
            elif height > 1000 or width > 1000:
                scale_factor = min(1000 / height, 1000 / width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return rgb_image
            
        except Exception as e:
            logger.error(f"PaddleOCR preprocessing failed: {e}")
            return image
    
    def _process_paddle_results(self, results: List, element_type: str) -> str:
        """PaddleOCR 결과 처리"""
        if not results or not results[0]:
            return ""
        
        # 모든 텍스트 추출
        text_lines = []
        confidences = []
        
        for line in results[0]:
            if len(line) >= 2:
                # line[0]: 좌표, line[1]: (텍스트, 신뢰도)
                text_info = line[1]
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    text, confidence = text_info[0], text_info[1]
                    if confidence > 0.5:  # 신뢰도 임계값
                        text_lines.append(text)
                        confidences.append(confidence)
        
        if not text_lines:
            return ""
        
        # 요소 타입에 따른 텍스트 조합
        if element_type == "title":
            # 제목은 보통 한 줄, 가장 신뢰도 높은 것 선택
            if confidences:
                max_conf_idx = confidences.index(max(confidences))
                return text_lines[max_conf_idx]
            return text_lines[0] if text_lines else ""
        
        elif element_type == "table":
            # 테이블은 탭으로 구분
            return "\t".join(text_lines)
        
        else:
            # 일반 텍스트는 줄바꿈으로 연결
            return "\n".join(text_lines)
    
    def _clean_ocr_text(self, text: str) -> str:
        """OCR 결과 텍스트 정리 (Tesseract 버전과 유사)"""
        if not text:
            return ""
        
        # 기본 정리
        cleaned = text.strip()
        
        # 연속된 공백을 단일 공백으로
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # PaddleOCR 특화 정리
        # 불필요한 특수문자 제거
        cleaned = re.sub(r'[^\w\s가-힣.,!?()[\]{}:;"\'`~@#$%^&*+=<>/-]', '', cleaned)
        
        # 한글과 영어 사이 공백 정리
        cleaned = re.sub(r'([가-힣])([A-Za-z])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([A-Za-z])([가-힣])', r'\1 \2', cleaned)
        
        return cleaned.strip()
    
    def get_detailed_results(self, image: np.ndarray) -> List[dict]:
        """상세한 OCR 결과 반환 (디버깅용)"""
        try:
            # Lazy initialization
            if not self._initialized:
                logger.info("Lazy loading PaddleOCR for detailed results...")
                self._initialize_paddle_ocr()
                self._initialized = True
                
            preprocessed_image = self._preprocess_image(image, "text")
            results = self.ocr_engine.ocr(preprocessed_image)
            
            detailed_results = []
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = line[1]  # (text, confidence)
                        
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text, confidence = text_info[0], text_info[1]
                            
                            detailed_results.append({
                                "text": text,
                                "confidence": confidence,
                                "bbox": bbox,
                                "bbox_simplified": {
                                    "x1": int(min(point[0] for point in bbox)),
                                    "y1": int(min(point[1] for point in bbox)),
                                    "x2": int(max(point[0] for point in bbox)),
                                    "y2": int(max(point[1] for point in bbox))
                                }
                            })
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"PaddleOCR detailed analysis failed: {e}")
            return []
    
    def set_language(self, lang: str):
        """언어 설정 변경"""
        if lang != self.lang:
            self.lang = lang
            self._initialized = False  # Force re-initialization on next use
            logger.info(f"Language changed to {lang}, will reinitialize PaddleOCR on next use")
    
    def compare_with_tesseract(self, image: np.ndarray, tesseract_service) -> dict:
        """Tesseract와 성능 비교"""
        try:
            # PaddleOCR 결과
            paddle_result = self.extract_text(image, "text")
            paddle_time_start = __import__('time').time()
            paddle_detailed = self.get_detailed_results(image)
            paddle_time = __import__('time').time() - paddle_time_start
            
            # Tesseract 결과
            tesseract_time_start = __import__('time').time()
            tesseract_result = tesseract_service.extract_text(image, "text")
            tesseract_time = __import__('time').time() - tesseract_time_start
            
            return {
                "paddle_ocr": {
                    "text": paddle_result,
                    "length": len(paddle_result),
                    "processing_time": paddle_time,
                    "word_count": len(paddle_result.split()) if paddle_result else 0,
                    "detailed_results_count": len(paddle_detailed)
                },
                "tesseract": {
                    "text": tesseract_result,
                    "length": len(tesseract_result),
                    "processing_time": tesseract_time,
                    "word_count": len(tesseract_result.split()) if tesseract_result else 0
                },
                "comparison": {
                    "paddle_longer": len(paddle_result) > len(tesseract_result),
                    "paddle_faster": paddle_time < tesseract_time,
                    "length_difference": len(paddle_result) - len(tesseract_result),
                    "time_difference": paddle_time - tesseract_time
                }
            }
            
        except Exception as e:
            logger.error(f"OCR comparison failed: {e}")
            return {"error": str(e)}