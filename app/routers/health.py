# ./app/routers/health.py
from fastapi import APIRouter, HTTPException
from app.services.llm_service import OllamaService
import logging

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)

@router.get("/ollama")
async def check_ollama_health():
    """Check Ollama service health and model availability"""
    try:
        service = OllamaService()  # ✅ Fixed: Use OllamaService instead of LLMService
        status = await service.check_ollama_status()
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_available_models():
    """List all available models in Ollama"""
    try:
        service = OllamaService()  # ✅ Fixed: Use OllamaService instead of LLMService
        models = await service.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pull-model/{model_name}")
async def pull_model(model_name: str):
    """Pull a specific model"""
    try:
        service = OllamaService()  # ✅ Fixed: Use OllamaService instead of LLMService
        success = await service.pull_model(model_name)
        if success:
            return {"message": f"Model {model_name} pulled successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to pull model {model_name}")
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
