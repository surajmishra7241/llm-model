from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AgentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    model: str = Field(..., description="Ollama model to use")
    system_prompt: str = Field(..., description="System prompt defining behavior")
    is_public: bool = Field(False)
    tools: List[str] = Field([])
    # Update to match the DB model
    metadata: Dict[str, Any] = Field({}, alias="agent_metadata")  # Use alias to match DB model

    class Config:
        populate_by_name = True  # Allows using both field name and alias

class AgentCreate(AgentBase):
    pass

class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    is_public: Optional[bool] = None
    tools: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="agent_metadata")

class AgentResponse(AgentBase):
    id: str
    owner_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True