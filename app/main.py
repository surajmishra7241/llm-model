from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import agents, auth, chat, rag, training, voice
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.metrics_middleware import metrics_middleware
from app.utils.health_check import router as health_router
from app.utils.logging import logger
from app.services.cache import cache_service
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await cache_service.connect()
    yield
    # Shutdown
    await cache_service.disconnect()

app = FastAPI(lifespan=lifespan)
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Multi-tenant AI agent platform with Ollama integration",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.middleware("http")(metrics_middleware)
    
    # Routers
    app.include_router(health_router)
    app.include_router(auth.router, prefix=settings.API_PREFIX)
    app.include_router(agents.router, prefix=settings.API_PREFIX)
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    app.include_router(rag.router, prefix=settings.API_PREFIX)
    app.include_router(training.router, prefix=f"{settings.API_PREFIX}/training")
    app.include_router(voice.router, prefix=f"{settings.API_PREFIX}/voice")
    
    @app.on_event("startup")
    async def startup():
        logger.info("Starting application...")
        await cache_service.connect()
        
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down application...")
        await cache_service.disconnect()
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None,
        access_log=False
    )