# ./app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import (
    agents, auth, chat, rag,
    training, voice, agent_interaction,
    health, execute
)
from app.middleware import LoggingMiddleware, RateLimiterMiddleware
import asyncio
import logging
from app.utils.health_check import router as health_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

if settings.RATE_LIMITING_ENABLED:
    app.add_middleware(RateLimiterMiddleware)

# Routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(agents.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(execute.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1/training")
app.include_router(health_router)
app.include_router(chat.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1/voice")
app.include_router(agent_interaction.router, prefix="/api/v1")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)