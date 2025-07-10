# ./app/models/agent_model.py
from datetime import datetime  # Add this import at the top
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PersonalityTrait(str, Enum):
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    WITTY = "witty"
    EMPATHETIC = "empathetic"
    ENTHUSIASTIC = "enthusiastic"

class EmotionalAwarenessConfig(BaseModel):
    detect_emotion: bool = True
    adjust_tone: bool = True
    empathy_level: float = Field(0.8, ge=0, le=1)
    max_emotional_response_time: float = Field(1.5, gt=0)

class MemoryConfig(BaseModel):
    context_window: int = Field(10, gt=0)
    long_term_memory: bool = False
    memory_refresh_interval: int = Field(300, gt=0)  # seconds

class AgentPersonality(BaseModel):
    traits: List[PersonalityTrait] = [PersonalityTrait.FRIENDLY]
    base_tone: str = "helpful and knowledgeable"
    emotional_awareness: EmotionalAwarenessConfig = EmotionalAwarenessConfig()
    memory: MemoryConfig = MemoryConfig()

class AgentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    model: str = Field(..., description="Ollama model to use")
    system_prompt: str = Field(..., description="System prompt defining behavior")
    is_public: bool = Field(False)
    tools: List[str] = Field([])
    personality: AgentPersonality = Field(default_factory=AgentPersonality)
    metadata: Optional[Dict[str, Any]] = Field(None, alias="agent_metadata")

class AgentCreate(AgentBase):
    pass

class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    is_public: Optional[bool] = None
    tools: Optional[List[str]] = None
    personality: Optional[AgentPersonality] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="agent_metadata")

class Agent(AgentBase):
    id: str
    owner_id: str
    rag_enabled: bool = False

class AgentResponse(AgentBase):
    id: str
    owner_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True




class AgentPermissions(str, Enum):
    EXECUTE = "execute"
    TRAIN = "train"
    SHARE = "share"
    MANAGE = "manage"

class AgentVisibility(str, Enum):
    PRIVATE = "private"
    TEAM = "team"
    ORGANIZATION = "organization"
    PUBLIC = "public"

class AgentRateLimit(BaseModel):
    requests_per_minute: int = Field(100, gt=0)
    concurrent_executions: int = Field(10, gt=0)

class AgentDeploymentConfig(BaseModel):
    gpu_required: bool = False
    memory_requirements: str = "1Gi"
    timeout_seconds: int = 300

class EnterpriseAgentConfig(BaseModel):
    visibility: AgentVisibility = AgentVisibility.PRIVATE
    rate_limits: AgentRateLimit = Field(default_factory=AgentRateLimit)
    deployment: AgentDeploymentConfig = Field(default_factory=AgentDeploymentConfig)
    allowed_domains: List[str] = []
    permissions: List[AgentPermissions] = [AgentPermissions.EXECUTE]