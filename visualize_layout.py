#!/usr/bin/env python3
"""
레이아웃 시각화 스크립트
PDF나 이미지 파일의 레이아웃 분석 결과를 시각화하여 저장하는 독립 실행 스크립트
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.services.layout_service import LayoutService
from app.services.ocr_service import OCRService
from app.utils.image_utils import ImageUtils
from app.utils.visualization_utils import LayoutVisualizer
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_and_visualize(
    input_path: str,
    output_dir: str = "output",
    confidence_threshold: float = 0.7,
    dpi: int = 200,
    page_number: int = None,
    save_json: bool = True,
    save_html: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True
):
    """
    파일을 분석하고 시각화 결과를 저장합니다.
    
    Args:
        input_path: 입력 파일 경로
        output_dir: 출력 디렉토리
        confidence_threshold: 신뢰도 임계값
        dpi: PDF DPI 설정
        page_number: PDF 특정 페이지 (None이면 모든 페이지)
        save_json: JSON 결과 저장 여부
        save_html: HTML 시각화 저장 여부
        show_labels: 라벨 표시 여부
        show_confidence: 신뢰도 표시 여부
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 파일 검증
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_path}")
    
    # 파일 이름 추출
    input_filename = Path(input_path).stem
    file_extension = Path(input_path).suffix.lower()
    
    # 서비스 초기화
    logger.info("모델 로딩 중...")
    layout_service = LayoutService(
        model_name=settings.layout_model_name,
        confidence_threshold=confidence_threshold
    )
    layout_service.load_model()
    
    ocr_service = OCRService(lang=settings.ocr_lang)
    visualizer = LayoutVisualizer()
    
    # 파일 타입에 따른 처리
    if file_extension == '.pdf':
        logger.info(f"PDF 파일 처리: {input_path}")
        
        with open(input_path, 'rb') as f:
            pdf_bytes = f.read()
        
        images = ImageUtils.pdf_to_images(pdf_bytes, dpi)
        
        if len(images) == 0:
            raise ValueError("PDF에서 페이지를 찾을 수 없습니다.")
        
        # 특정 페이지만 처리하거나 모든 페이지 처리
        page_range = [page_number] if page_number else range(1, len(images) + 1)
        
        all_results = []
        
        for page_idx, page_num in enumerate(page_range):
            if page_num > len(images):
                logger.warning(f"페이지 {page_num}이 존재하지 않습니다. 총 {len(images)}페이지")
                continue
                
            logger.info(f"페이지 {page_num}/{len(images)} 처리 중...")
            
            image = images[page_num - 1]
            image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
            width, height = ImageUtils.get_image_dimensions(image)
            
            # 레이아웃 감지
            elements_data = layout_service.detect(image, confidence_threshold)
            
            # OCR 수행
            elements = []
            for element_data in elements_data:
                bbox_coords = element_data['coordinates']
                cropped_image = ImageUtils.crop_element(image, bbox_coords)
                
                content = None
                element_type = element_data['type']
                
                if element_type in ['text', 'title', 'list', 'table']:
                    content = ocr_service.extract_text(cropped_image, element_type)
                
                layout_element = {
                    'element_id': element_data['id'],
                    'element_type': element_type,
                    'bounding_box': element_data['bbox'],
                    'confidence_score': element_data['confidence'],
                    'text_content': content
                }
                
                elements.append(layout_element)
            
            # 결과 저장
            page_suffix = f"_page_{page_num}" if len(page_range) > 1 else ""
            
            # 이미지 시각화 저장
            output_image_path = os.path.join(output_dir, f"{input_filename}{page_suffix}_visualization.png")
            visualizer.save_visualization(image, elements, output_image_path, include_legend=True)
            logger.info(f"시각화 이미지 저장: {output_image_path}")
            
            # JSON 결과 저장
            if save_json:
                result_data = {
                    'input_file': input_path,
                    'page': page_num,
                    'total_pages': len(images),
                    'image_width': width,
                    'image_height': height,
                    'confidence_threshold': confidence_threshold,
                    'total_elements': len(elements),
                    'elements': elements
                }
                
                json_path = os.path.join(output_dir, f"{input_filename}{page_suffix}_result.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                logger.info(f"JSON 결과 저장: {json_path}")
                
                all_results.append(result_data)
            
            # HTML 시각화 저장
            if save_html:
                html_title = f"{input_filename} 페이지 {page_num} 레이아웃 분석"
                html_content = visualizer.create_html_visualization(image, elements, html_title)
                
                html_path = os.path.join(output_dir, f"{input_filename}{page_suffix}_visualization.html")
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML 시각화 저장: {html_path}")
        
        # 전체 페이지 요약 저장
        if save_json and len(all_results) > 1:
            summary_data = {
                'input_file': input_path,
                'total_pages': len(images),
                'processed_pages': len(all_results),
                'confidence_threshold': confidence_threshold,
                'total_elements_all_pages': sum(r['total_elements'] for r in all_results),
                'pages': all_results
            }
            
            summary_path = os.path.join(output_dir, f"{input_filename}_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            logger.info(f"요약 저장: {summary_path}")
    
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
        logger.info(f"이미지 파일 처리: {input_path}")
        
        # 이미지 로드
        with open(input_path, 'rb') as f:
            image_bytes = f.read()
        
        image = ImageUtils.load_image_from_bytes(image_bytes)
        image = ImageUtils.resize_image_if_needed(image, settings.max_image_size)
        width, height = ImageUtils.get_image_dimensions(image)
        
        # 레이아웃 감지
        elements_data = layout_service.detect(image, confidence_threshold)
        
        # OCR 수행
        elements = []
        for element_data in elements_data:
            bbox_coords = element_data['coordinates']
            cropped_image = ImageUtils.crop_element(image, bbox_coords)
            
            content = None
            element_type = element_data['type']
            
            if element_type in ['text', 'title', 'list', 'table']:
                content = ocr_service.extract_text(cropped_image, element_type)
            
            layout_element = {
                'element_id': element_data['id'],
                'element_type': element_type,
                'bounding_box': element_data['bbox'],
                'confidence_score': element_data['confidence'],
                'text_content': content
            }
            
            elements.append(layout_element)
        
        # 결과 저장
        # 이미지 시각화 저장
        output_image_path = os.path.join(output_dir, f"{input_filename}_visualization.png")
        visualizer.save_visualization(image, elements, output_image_path, include_legend=True)
        logger.info(f"시각화 이미지 저장: {output_image_path}")
        
        # JSON 결과 저장
        if save_json:
            result_data = {
                'input_file': input_path,
                'image_width': width,
                'image_height': height,
                'confidence_threshold': confidence_threshold,
                'total_elements': len(elements),
                'elements': elements
            }
            
            json_path = os.path.join(output_dir, f"{input_filename}_result.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON 결과 저장: {json_path}")
        
        # HTML 시각화 저장
        if save_html:
            html_title = f"{input_filename} 레이아웃 분석"
            html_content = visualizer.create_html_visualization(image, elements, html_title)
            
            html_path = os.path.join(output_dir, f"{input_filename}_visualization.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML 시각화 저장: {html_path}")
    
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_extension}")
    
    logger.info("처리 완료!")

def main():
    parser = argparse.ArgumentParser(
        description="PDF/이미지 레이아웃 분석 및 시각화 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 이미지 분석
  python visualize_layout.py sample.png

  # PDF 전체 페이지 분석
  python visualize_layout.py document.pdf

  # PDF 특정 페이지만 분석
  python visualize_layout.py document.pdf --page 3

  # 출력 디렉토리 지정
  python visualize_layout.py sample.png --output results/

  # 신뢰도 임계값 조정
  python visualize_layout.py sample.png --confidence 0.8

  # 라벨 없이 시각화
  python visualize_layout.py sample.png --no-labels
        """
    )
    
    parser.add_argument(
        'input_file',
        help='분석할 PDF 또는 이미지 파일'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='출력 디렉토리 (기본값: output)'
    )
    
    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.7,
        help='신뢰도 임계값 (0.1-0.99, 기본값: 0.7)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='PDF DPI 설정 (100-400, 기본값: 200)'
    )
    
    parser.add_argument(
        '--page',
        type=int,
        help='PDF에서 처리할 특정 페이지 번호'
    )
    
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='JSON 결과 저장 안함'
    )
    
    parser.add_argument(
        '--no-html',
        action='store_true',
        help='HTML 시각화 저장 안함'
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='시각화에서 라벨 표시 안함'
    )
    
    parser.add_argument(
        '--no-confidence',
        action='store_true',
        help='시각화에서 신뢰도 표시 안함'
    )
    
    args = parser.parse_args()
    
    # 인자 검증
    if args.confidence < 0.1 or args.confidence > 0.99:
        parser.error("신뢰도는 0.1에서 0.99 사이여야 합니다.")
    
    if args.dpi < 100 or args.dpi > 400:
        parser.error("DPI는 100에서 400 사이여야 합니다.")
    
    try:
        analyze_and_visualize(
            input_path=args.input_file,
            output_dir=args.output,
            confidence_threshold=args.confidence,
            dpi=args.dpi,
            page_number=args.page,
            save_json=not args.no_json,
            save_html=not args.no_html,
            show_labels=not args.no_labels,
            show_confidence=not args.no_confidence
        )
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()