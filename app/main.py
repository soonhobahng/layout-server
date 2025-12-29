from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
import json

from .api.routes import router
from .services.layout_service import LayoutService
from .services.ocr_factory import OCRFactory
from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting layout analysis server...")
    
    try:
        logger.info("Loading layout detection model...")
        layout_service = LayoutService(
            model_name=settings.layout_model_name,
            confidence_threshold=settings.confidence_threshold
        )
        layout_service.load_model()
        app.state.layout_service = layout_service
        
        logger.info("Initializing OCR service...")
        ocr_service = OCRFactory.create_ocr_service()
        app.state.ocr_service = ocr_service
        
        # OCR 엔진 정보 로그
        engine_info = OCRFactory.get_engine_info()
        logger.info(f"OCR engine: {engine_info['current_engine']}")
        logger.info(f"Available engines: {engine_info['available_engines']}")
        
        logger.info("Server startup complete")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise HTTPException(status_code=503, detail=f"Service initialization failed: {str(e)}")
    
    yield
    
    logger.info("Shutting down server...")

app = FastAPI(
    title="Layout Analysis Server",
    description="FastAPI server for PDF/Image layout analysis using layoutparser and OCR",
    version="1.0.0",
    lifespan=lifespan
)

def custom_json_encoder(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PrettyJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=settings.json_ensure_ascii,
            indent=settings.json_indent,
            sort_keys=settings.json_sort_keys,
            default=custom_json_encoder
        ).encode("utf-8")

app.include_router(router, default_response_class=PrettyJSONResponse)

@app.get("/", response_class=PrettyJSONResponse)
async def root():
    return {
        "message": "Layout Analysis Server", 
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )