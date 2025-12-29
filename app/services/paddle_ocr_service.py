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
            
            # 한국어 + 영어 지원 (새로운 API 사용)
            if self.lang == "korean":
                self.ocr_engine = PaddleOCR(
                    use_textline_orientation=True,  # use_angle_cls 대신 사용
                    lang='korean'
                )
            else:
                self.ocr_engine = PaddleOCR(
                    use_textline_orientation=True,  # use_angle_cls 대신 사용
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
            logger.info(f"OCR extract_text called - element_type: {element_type}, image shape: {image.shape}")
            
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
            
            # PaddleOCR 실행 (새로운 API 사용)
            results = self.ocr_engine.predict(preprocessed_image)
            
            # 결과 처리 (새로운 구조에 맞게)
            extracted_text = self._process_paddle_results_v3(results, element_type)
            
            # 텍스트 후처리
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            logger.debug(f"OCR result for {element_type}: '{cleaned_text}' (length: {len(cleaned_text)})")
            
            return cleaned_text if cleaned_text else ""
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            logger.error(f"Image shape: {image.shape}, Element type: {element_type}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
    
    def _process_paddle_results_v3(self, results: List, element_type: str) -> str:
        """PaddleOCR 3.3.x 결과 처리 (새로운 구조)"""
        try:
            if not results or len(results) == 0:
                logger.warning(f"No OCR results for element type: {element_type}")
                return ""
            
            # 새로운 구조에서 텍스트 추출
            result = results[0] if isinstance(results, list) else results
            
            # rec_texts 필드에서 텍스트 추출
            if isinstance(result, dict) and 'rec_texts' in result:
                texts = result['rec_texts']
                scores = result.get('rec_scores', [])
                
                logger.info(f"Found {len(texts)} text segments")
                
                # 신뢰도 필터링
                filtered_texts = []
                for i, text in enumerate(texts):
                    score = scores[i] if i < len(scores) else 1.0
                    if score > 0.3:  # 신뢰도 임계값
                        filtered_texts.append(text)
                        logger.debug(f"Accepted text: '{text}' (score: {score:.3f})")
                    else:
                        logger.debug(f"Rejected text: '{text}' (score: {score:.3f})")
                
                if not filtered_texts:
                    logger.warning(f"No text passed confidence filter for {element_type}")
                    return ""
                
                # 요소 타입에 따른 조합
                if element_type == "title":
                    return filtered_texts[0] if filtered_texts else ""
                elif element_type == "table":
                    return "\t".join(filtered_texts)
                else:
                    return "\n".join(filtered_texts)
            else:
                logger.warning(f"Unexpected result structure: {type(result)}")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing PaddleOCR v3 results: {e}")
            return ""
    
    def _process_paddle_results(self, results: List, element_type: str) -> str:
        """PaddleOCR 결과 처리 (구 버전 호환성)"""
        if not results or not results[0]:
            logger.warning(f"No OCR results for element type: {element_type}")
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
                    if confidence > 0.3:  # 신뢰도 임계값을 낮춤 (0.5 -> 0.3)
                        text_lines.append(text)
                        confidences.append(confidence)
                        logger.debug(f"OCR detected: '{text}' (confidence: {confidence:.3f})")
                    else:
                        logger.debug(f"OCR rejected low confidence: '{text}' (confidence: {confidence:.3f})")
        
        if not text_lines:
            logger.warning(f"No text extracted after confidence filtering for element type: {element_type}")
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
            results = self.ocr_engine.predict(preprocessed_image)  # 새로운 API 사용
            
            detailed_results = []
            if results and len(results) > 0:
                result = results[0] if isinstance(results, list) else results
                
                if isinstance(result, dict):
                    texts = result.get('rec_texts', [])
                    scores = result.get('rec_scores', [])
                    polys = result.get('rec_polys', [])
                    
                    for i, text in enumerate(texts):
                        score = scores[i] if i < len(scores) else 1.0
                        bbox = polys[i] if i < len(polys) else []
                        
                        if bbox and len(bbox) >= 4:
                            detailed_results.append({
                                "text": text,
                                "confidence": score,
                                "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                                "bbox_simplified": {
                                    "x1": int(min(point[0] for point in bbox)),
                                    "y1": int(min(point[1] for point in bbox)),
                                    "x2": int(max(point[0] for point in bbox)),
                                    "y2": int(max(point[1] for point in bbox))
                                } if len(bbox) > 0 else {}
                            })
                        else:
                            detailed_results.append({
                                "text": text,
                                "confidence": score,
                                "bbox": [],
                                "bbox_simplified": {}
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
    
    def _perform_ocr(self, image: np.ndarray, element_type: str) -> str:
        """OCR 실행 (디버깅 API용)"""
        try:
            if not self._initialized:
                self._initialize_paddle_ocr()
                self._initialized = True
                
            results = self.ocr_engine.predict(image)  # 새로운 API 사용
            return self._process_paddle_results_v3(results, element_type)  # 새로운 처리 함수 사용
        except Exception as e:
            logger.error(f"_perform_ocr failed: {e}")
            return ""
    
    def _advanced_preprocess(self, image: np.ndarray, element_type: str) -> np.ndarray:
        """향상된 전처리 (디버깅 API용)"""
        try:
            # 기본 전처리 + 추가 향상
            preprocessed = self._preprocess_image(image, element_type)
            
            # 추가 대비 향상
            if len(preprocessed.shape) == 3:
                gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
            else:
                gray = preprocessed
                
            # 적응형 히스토그램 균등화
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # RGB로 다시 변환
            if len(preprocessed.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            return enhanced
        except Exception as e:
            logger.error(f"Advanced preprocessing failed: {e}")
            return image
    
    def _scale_first_preprocess(self, image: np.ndarray, element_type: str) -> np.ndarray:
        """스케일 우선 전처리 (디버깅 API용)"""
        try:
            # 4배 확대 후 전처리
            height, width = image.shape[:2]
            scaled = cv2.resize(image, (width*4, height*4), interpolation=cv2.INTER_CUBIC)
            return self._preprocess_image(scaled, element_type)
        except Exception as e:
            logger.error(f"Scale-first preprocessing failed: {e}")
            return image
    
    def _fallback_ocr(self, image: np.ndarray, element_type: str) -> str:
        """Fallback OCR (디버깅 API용)"""
        try:
            # 여러 전처리 방식으로 시도
            methods = [
                lambda img: img,  # 원본
                lambda img: self._preprocess_image(img, element_type),  # 기본 전처리
                lambda img: self._advanced_preprocess(img, element_type),  # 향상된 전처리
                lambda img: self._scale_first_preprocess(img, element_type)  # 스케일 우선
            ]
            
            best_result = ""
            for method in methods:
                try:
                    processed = method(image)
                    result = self._perform_ocr(processed, element_type)
                    if len(result) > len(best_result):
                        best_result = result
                except:
                    continue
            
            return best_result
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return ""