import layoutparser as lp
import numpy as np
from typing import List, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)

class MockLayoutElement:
    def __init__(self, x1, y1, x2, y2, element_type, score=0.8):
        self.coordinates = (x1, y1, x2, y2)
        self.type = element_type
        self.score = score

class MockLayoutModel:
    def __init__(self):
        logger.info("Using mock layout model for development")
    
    def detect(self, image):
        # Return mock layout elements for testing
        height, width = image.shape[:2]
        return [
            MockLayoutElement(10, 10, width//2-10, 100, "title", 0.95),
            MockLayoutElement(10, 120, width-10, height//2, "text", 0.85),
            MockLayoutElement(10, height//2+10, width-10, height-10, "text", 0.80)
        ]

class LayoutService:
    def __init__(self, model_name: str, confidence_threshold: float = 0.7):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.label_map = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
        self.model = None
        
    def load_model(self):
        try:
            config_path = f'lp://PubLayNet/{self.model_name}/config'
            logger.info(f"Loading layout model: {config_path}")
            
            # Try Detectron2 model first
            try:
                self.model = lp.Detectron2LayoutModel(
                    config_path=config_path,
                    label_map=self.label_map,
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.confidence_threshold]
                )
                logger.info("Detectron2 layout model loaded successfully")
                return
            except (AttributeError, ImportError) as e:
                logger.warning(f"Detectron2 model unavailable: {e}")
            
            # Fallback to basic layout model if available
            try:
                self.model = lp.models.Detectron2LayoutModel(
                    config_path=config_path,
                    label_map=self.label_map,
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.confidence_threshold]
                )
                logger.info("Basic layout model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Basic model also unavailable: {e}")
            
            # Last resort: create a mock model for development
            logger.warning("No layout models available, using mock model for testing")
            self.model = MockLayoutModel()
            
        except Exception as e:
            logger.error(f"Failed to load layout model: {e}")
            raise
    
    def detect(self, image: np.ndarray, custom_threshold: float = None) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            threshold = custom_threshold or self.confidence_threshold
            
            if custom_threshold is not None and custom_threshold != self.confidence_threshold:
                config_path = f'lp://PubLayNet/{self.model_name}/config'
                temp_model = lp.Detectron2LayoutModel(
                    config_path=config_path,
                    label_map=self.label_map,
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", custom_threshold]
                )
                layout_result = temp_model.detect(image)
            else:
                layout_result = self.model.detect(image)
            
            elements = []
            for idx, element in enumerate(layout_result):
                x1, y1, x2, y2 = element.coordinates
                elements.append({
                    'id': idx + 1,
                    'type': element.type,
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    },
                    'confidence': float(element.score) if hasattr(element, 'score') else threshold,
                    'coordinates': (int(x1), int(y1), int(x2), int(y2))
                })
            
            return elements
            
        except Exception as e:
            logger.error(f"Layout detection failed: {e}")
            raise
    
    def is_gpu_available(self) -> bool:
        return torch.cuda.is_available()
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'gpu_available': self.is_gpu_available(),
            'label_map': self.label_map
        }