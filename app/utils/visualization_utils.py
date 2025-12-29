import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
import colorsys
import random

class LayoutVisualizer:
    def __init__(self):
        self.colors = self._generate_colors()
        self.type_colors = {
            'text': '#FF6B6B',      # 빨간색
            'title': '#4ECDC4',     # 청록색  
            'list': '#45B7D1',      # 파란색
            'table': '#96CEB4',     # 녹색
            'figure': '#FFEAA7'     # 노란색
        }
        
    def _generate_colors(self) -> List[str]:
        colors = []
        for i in range(20):
            hue = i / 20.0
            saturation = 0.7 + (i % 3) * 0.1
            lightness = 0.5 + (i % 2) * 0.2
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _get_element_color(self, element_type: str, element_id: int) -> Tuple[int, int, int]:
        if element_type in self.type_colors:
            return self._hex_to_rgb(self.type_colors[element_type])
        else:
            color_idx = element_id % len(self.colors)
            return self._hex_to_rgb(self.colors[color_idx])
    
    def visualize_layout(
        self, 
        image: np.ndarray, 
        elements: List[Dict[str, Any]], 
        show_labels: bool = True,
        show_confidence: bool = True,
        box_thickness: int = 2,
        font_size: int = 20
    ) -> np.ndarray:
        try:
            # 입력 이미지 검증
            if image is None or image.size == 0:
                raise ValueError("Empty or None image provided")
            
            # numpy array를 PIL Image로 변환
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 이미지 형태에 따른 변환
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # BGR to RGB 변환
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                elif image.shape[2] == 4:
                    # BGRA to RGBA 변환
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
                else:
                    pil_image = Image.fromarray(image)
            elif len(image.shape) == 2:
                # 그레이스케일
                pil_image = Image.fromarray(image, mode='L').convert('RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
                
        except Exception as e:
            print(f"Error in image conversion: {e}")
            # 기본 이미지 생성
            pil_image = Image.new('RGB', (800, 600), 'white')
        
        # 투명 레이어 생성
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 폰트 설정
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        for element in elements:
            # 경계 박스 좌표
            bbox = element.get('bounding_box', element.get('bbox', {}))
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # 요소 정보
            element_type = element.get('element_type', element.get('type', 'unknown'))
            element_id = element.get('element_id', element.get('id', 0))
            confidence = element.get('confidence_score', element.get('confidence', 0.0))
            content = element.get('text_content', element.get('content', ''))
            
            # 색상 선택
            color = self._get_element_color(element_type, element_id)
            
            # 경계 박스 그리기 (반투명)
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color + (255,),
                fill=color + (50,),
                width=box_thickness
            )
            
            # 라벨 표시
            if show_labels:
                # 라벨 텍스트 구성
                label_parts = [f"{element_type}"]
                if show_confidence:
                    label_parts.append(f"{confidence:.2f}")
                
                label = f"[{element_id}] " + " | ".join(label_parts)
                
                # 텍스트 크기 계산
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # 라벨 배경
                label_x = x1
                label_y = max(0, y1 - text_height - 5)
                
                draw.rectangle(
                    [label_x, label_y, label_x + text_width + 10, label_y + text_height + 5],
                    fill=color + (200,)
                )
                
                # 라벨 텍스트
                draw.text(
                    (label_x + 5, label_y + 2),
                    label,
                    fill=(255, 255, 255, 255),
                    font=font
                )
                
                # 내용 미리보기 (짧은 텍스트만)
                if content and len(content.strip()) > 0:
                    preview = content.strip()[:30] + "..." if len(content.strip()) > 30 else content.strip()
                    preview = preview.replace('\n', ' ')
                    
                    content_bbox = draw.textbbox((0, 0), preview, font=font)
                    content_width = content_bbox[2] - content_bbox[0]
                    content_height = content_bbox[3] - content_bbox[1]
                    
                    content_x = x1
                    content_y = y2 + 5
                    
                    # 이미지 범위를 벗어나지 않도록 조정
                    if content_y + content_height + 5 > pil_image.height:
                        content_y = y1 - content_height - 30
                    
                    if content_x + content_width + 10 > pil_image.width:
                        content_x = pil_image.width - content_width - 15
                    
                    draw.rectangle(
                        [content_x, content_y, content_x + content_width + 10, content_y + content_height + 5],
                        fill=(0, 0, 0, 180)
                    )
                    
                    draw.text(
                        (content_x + 5, content_y + 2),
                        preview,
                        fill=(255, 255, 255, 255),
                        font=font
                    )
        
        # 원본 이미지와 오버레이 합성
        try:
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
            
            result = Image.alpha_composite(pil_image, overlay)
            
            # numpy array로 다시 변환
            result_array = np.array(result.convert('RGB'))
            
            # 배열 크기 검증
            if len(result_array.shape) != 3:
                raise ValueError(f"Invalid result array shape: {result_array.shape}")
            
            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error in image composition: {e}")
            # 오류 시 원본 이미지 반환 시도
            try:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    return image
                else:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
            except:
                # 최종 폴백: 기본 이미지
                return np.full((600, 800, 3), 255, dtype=np.uint8)
    
    def create_legend(self, elements: List[Dict[str, Any]], width: int = 300, height: int = None) -> np.ndarray:
        # 요소 타입별 개수 계산
        type_counts = {}
        for element in elements:
            element_type = element.get('element_type', element.get('type', 'unknown'))
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        # 요소가 없으면 기본 범례 생성
        if not type_counts:
            type_counts = {'text': 0}
        
        # 동적 높이 계산
        if height is None:
            num_items = len(type_counts)
            height = max(200, 40 + num_items * 25 + 30)  # 최소 200px, 항목당 25px
        
        # 범례 이미지 생성
        legend_img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(legend_img)
        
        # 폰트 로딩 (더 안전한 방법)
        try:
            # macOS
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            try:
                # Windows
                font = ImageFont.truetype("arial.ttf", 16)
                title_font = ImageFont.truetype("arial.ttf", 20)
            except:
                try:
                    # Linux
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
                    title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
                except:
                    # 기본 폰트
                    font = ImageFont.load_default()
                    title_font = ImageFont.load_default()
        
        # 제목
        draw.text((10, 10), "레이아웃 요소", fill='black', font=title_font)
        
        # 요소별 범례 그리기
        y_offset = 40
        for i, (element_type, count) in enumerate(type_counts.items()):
            if y_offset > height - 30:
                break
                
            color = self._get_element_color(element_type, 0)
            
            # 색상 박스
            draw.rectangle([15, y_offset, 35, y_offset + 15], fill=color, outline='black')
            
            # 텍스트
            type_names = {
                'text': '텍스트',
                'title': '제목', 
                'list': '목록',
                'table': '표',
                'figure': '그림'
            }
            display_name = type_names.get(element_type, element_type)
            text = f"{display_name} ({count}개)" if count > 0 else display_name
            draw.text((45, y_offset), text, fill='black', font=font)
            
            y_offset += 25
        
        # numpy 배열로 변환 시 오류 처리
        try:
            legend_array = np.array(legend_img)
            if len(legend_array.shape) == 3:
                return cv2.cvtColor(legend_array, cv2.COLOR_RGB2BGR)
            else:
                # 그레이스케일인 경우
                return cv2.cvtColor(legend_array, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Warning: Legend conversion failed: {e}")
            # 기본 범례 생성
            default_legend = np.full((height, width, 3), 255, dtype=np.uint8)
            cv2.putText(default_legend, "Legend Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            return default_legend
    
    def save_visualization(
        self, 
        image: np.ndarray, 
        elements: List[Dict[str, Any]], 
        output_path: str,
        include_legend: bool = True
    ):
        # 시각화된 이미지 생성
        visualized = self.visualize_layout(image, elements)
        
        if include_legend:
            # 범례 생성
            legend = self.create_legend(elements)
            
            # 이미지와 범례 크기 맞추기
            vis_height, vis_width = visualized.shape[:2]
            legend_height, legend_width = legend.shape[:2]
            
            # 범례 크기를 메인 이미지 높이에 맞춤
            if legend_height != vis_height:
                # 비율을 유지하면서 높이를 맞춤
                aspect_ratio = legend_width / legend_height
                new_width = int(vis_height * aspect_ratio)
                legend_resized = cv2.resize(legend, (new_width, vis_height))
            else:
                legend_resized = legend
            
            # 두 이미지를 수평으로 결합
            try:
                combined = np.hstack([visualized, legend_resized])
                cv2.imwrite(output_path, combined)
            except ValueError as e:
                # 크기가 맞지 않으면 별도로 저장
                print(f"Warning: Could not combine images due to size mismatch: {e}")
                cv2.imwrite(output_path, visualized)
                legend_path = output_path.replace('.png', '_legend.png')
                cv2.imwrite(legend_path, legend_resized)
                print(f"Legend saved separately to: {legend_path}")
        else:
            cv2.imwrite(output_path, visualized)
    
    def to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def create_html_visualization(
        self, 
        image: np.ndarray, 
        elements: List[Dict[str, Any]], 
        title: str = "레이아웃 분석 결과"
    ) -> str:
        # 시각화된 이미지 생성
        visualized = self.visualize_layout(image, elements)
        img_base64 = self.to_base64(visualized)
        
        # 범례 생성
        legend = self.create_legend(elements)
        legend_base64 = self.to_base64(legend)
        
        # HTML 생성
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .visualization {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .main-image {{
                    flex: 1;
                }}
                .legend {{
                    width: 300px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                .stat-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .stat-label {{
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>문서 레이아웃 분석 및 요소 검출 결과</p>
                </div>
                
                <div class="visualization">
                    <div class="main-image">
                        <img src="{img_base64}" alt="레이아웃 분석 결과">
                    </div>
                    <div class="legend">
                        <img src="{legend_base64}" alt="범례">
                    </div>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(elements)}</div>
                        <div class="stat-label">총 요소 수</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len([e for e in elements if e.get('element_type', e.get('type')) == 'text'])}</div>
                        <div class="stat-label">텍스트 영역</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len([e for e in elements if e.get('element_type', e.get('type')) == 'table'])}</div>
                        <div class="stat-label">표</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len([e for e in elements if e.get('element_type', e.get('type')) == 'figure'])}</div>
                        <div class="stat-label">그림</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template