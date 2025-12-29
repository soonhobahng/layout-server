import pytesseract
import numpy as np
from typing import Optional, List
import logging
import cv2
from PIL import Image, ImageEnhance

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
            
            # 다중 전처리 방식으로 최적 결과 찾기
            results = []
            
            # 1. 기본 전처리
            preprocessed1 = self._preprocess_image(image, element_type)
            text1 = self._perform_ocr(preprocessed1, element_type)
            results.append(text1)
            
            # 2. 향상된 전처리 (대비, 선명도 개선)
            preprocessed2 = self._advanced_preprocess(image, element_type)
            text2 = self._perform_ocr(preprocessed2, element_type)
            results.append(text2)
            
            # 3. 스케일링 우선 전처리
            preprocessed3 = self._scale_first_preprocess(image, element_type)
            text3 = self._perform_ocr(preprocessed3, element_type)
            results.append(text3)
            
            # 결과 정리 및 최적화
            cleaned_results = []
            for result in results:
                cleaned = self._clean_ocr_text(result)
                if cleaned:
                    cleaned_results.append(cleaned)
            
            # 가장 적절한 결과 선택
            best_result = self._select_best_result(cleaned_results)
            
            # 결과가 여전히 부족하면 fallback 시도
            if len(best_result) < 3:
                fallback_result = self._fallback_ocr(image, element_type)
                fallback_cleaned = self._clean_ocr_text(fallback_result)
                if len(fallback_cleaned) > len(best_result):
                    best_result = fallback_cleaned
            
            return best_result
            
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
    
    def _perform_ocr(self, image: np.ndarray, element_type: str) -> str:
        """OCR 실행"""
        try:
            config = self._get_tesseract_config(element_type)
            text = pytesseract.image_to_string(
                image, 
                lang=self.lang, 
                config=config
            ).strip()
            return text
        except Exception as e:
            logger.error(f"OCR execution failed: {e}")
            return ""
    
    def _advanced_preprocess(self, image: np.ndarray, element_type: str) -> np.ndarray:
        """향상된 이미지 전처리"""
        try:
            # numpy to PIL 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            # PIL to numpy 변환
            enhanced = np.array(pil_image)
            
            # 그레이스케일 변환
            if len(enhanced.shape) == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced
            
            # 적응적 임계값
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 크기 조정
            height, width = binary.shape
            if height < 100 or width < 100:
                scale = max(100 / height, 100 / width, 3.0)
                new_height = int(height * scale)
                new_width = int(width * scale)
                binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return binary
            
        except Exception as e:
            logger.error(f"Advanced preprocessing failed: {e}")
            return self._preprocess_image(image, element_type)
    
    def _scale_first_preprocess(self, image: np.ndarray, element_type: str) -> np.ndarray:
        """스케일링 우선 전처리"""
        try:
            # 먼저 크기 확대
            height, width = image.shape[:2]
            scale_factor = 4.0  # 4배 확대
            
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            if len(image.shape) == 3:
                upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 가우시안 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # OTSU 임계값
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 모폴로지 연산으로 텍스트 개선
            if element_type == "text":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
            
        except Exception as e:
            logger.error(f"Scale-first preprocessing failed: {e}")
            return self._preprocess_image(image, element_type)
    
    def _fallback_ocr(self, image: np.ndarray, element_type: str) -> str:
        """다른 PSM 모드로 fallback OCR 시도"""
        try:
            preprocessed = self._preprocess_image(image, element_type)
            
            # 여러 PSM 모드 시도
            psm_modes = [8, 7, 13, 6, 4] if element_type == "title" else [6, 8, 7, 13, 4]
            
            results = []
            
            for psm in psm_modes:
                try:
                    config = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(
                        preprocessed,
                        lang=self.lang,
                        config=config
                    ).strip()
                    if text:
                        results.append(text)
                except:
                    continue
            
            return max(results, key=len) if results else ""
            
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return ""
    
    def _get_tesseract_config(self, element_type: str) -> str:
        """요소 타입별 Tesseract 설정"""
        # 간단한 설정으로 변경 (문자 화이트리스트 제거)
        if element_type == "title":
            return '--oem 3 --psm 8'
        elif element_type == "text":
            return '--oem 3 --psm 6' 
        elif element_type == "list":
            return '--oem 3 --psm 6'
        elif element_type == "table":
            return '--oem 3 --psm 6'
        else:
            return '--oem 3 --psm 6'
    
    def _clean_ocr_text(self, text: str) -> str:
        """OCR 결과 텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        cleaned = text.strip()
        
        # 연속된 공백을 단일 공백으로
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 잘못 인식된 문자 패턴 수정
        replacements = {
            # 일반적인 OCR 오류 패턴
            'ㅇ': '',  # 단독 ㅇ 제거
            'ㅁ': '',  # 단독 ㅁ 제거
            'ㅂ': '',  # 단독 ㅂ 제거
            '|': 'l',  # 파이프를 소문자 l로
            '0': 'O',  # 0을 O로 (문맥에 따라)
        }
        
        # 의미있는 길이의 텍스트만 처리
        if len(cleaned) > 2:
            for old, new in replacements.items():
                if old in cleaned and len(old) == 1:  # 단일 문자 오류만 수정
                    cleaned = cleaned.replace(old, new)
        
        # 한글과 영어가 섞인 경우 공백 정리
        cleaned = re.sub(r'([가-힣])([A-Za-z])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([A-Za-z])([가-힣])', r'\1 \2', cleaned)
        
        return cleaned.strip()
    
    def _select_best_result(self, results: List[str]) -> str:
        """여러 OCR 결과 중 최적 결과 선택"""
        if not results:
            return ""
        
        if len(results) == 1:
            return results[0]
        
        # 점수 기반 선택
        scored_results = []
        for result in results:
            score = self._calculate_text_quality_score(result)
            scored_results.append((result, score))
        
        # 가장 높은 점수 선택
        best_result = max(scored_results, key=lambda x: x[1])
        return best_result[0]
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """텍스트 품질 점수 계산"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # 기본 점수: 길이
        score += len(text) * 0.1
        
        # 한글 포함 시 가산점
        import re
        if re.search(r'[가-힣]', text):
            score += 10
        
        # 영어 포함 시 가산점
        if re.search(r'[A-Za-z]', text):
            score += 5
        
        # 숫자 포함 시 가산점
        if re.search(r'\d', text):
            score += 3
        
        # 특수문자가 너무 많으면 감점
        special_chars = len(re.findall(r'[^\w\s가-힣]', text))
        if special_chars > len(text) * 0.3:
            score -= special_chars * 2
        
        # 연속된 같은 문자가 많으면 감점 (OCR 오류 가능성)
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        score -= repeated_chars * 5
        
        return score
    
    def set_language(self, lang: str):
        self.lang = lang