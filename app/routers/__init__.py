# ./app/routers/__init__.py
from .auth import router as auth_router
from .agents import router as agents_router
from .chat import router as chat_router
from .rag import router as rag_router
from .training import router as training_router
from .voice import router as voice_router
from .agent_interaction import router as agent_interaction_router
from .health import router as health_router
from .execute import router as execute_router
from .monitoring import router as monitoring_router

__all__ = [
    "auth_router",
    "agents_router",
    "chat_router",
    "rag_router",
    "training_router",
    "voice_router",
    "agent_interaction_router",
    "health_router",
    "execute_router",
    "monitoring_router"
]