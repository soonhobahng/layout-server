from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    layout_model_name: str = "faster_rcnn_R_50_FPN_3x"
    confidence_threshold: float = 0.7
    max_image_size: int = 4096
    ocr_lang: str = "kor+eng"
    
    json_ensure_ascii: bool = False
    json_indent: int = 2
    json_sort_keys: bool = True
    
    model_config = {"protected_namespaces": ("settings_",)}

settings = Settings()