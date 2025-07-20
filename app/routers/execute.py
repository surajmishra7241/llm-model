from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from app.dependencies import get_current_user, get_db
from app.services.agent_execution_service import execute_agent
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import time

router = APIRouter(prefix="/execute", tags=["Agent Execution"])
logger = logging.getLogger(__name__)

# Enhanced request models
class ExecutionParameters(BaseModel):
    """Enhanced parameters for agent execution"""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2000, ge=1, le=8000)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    
    # Search parameters
    enable_search: Optional[bool] = True
    search_sources: Optional[List[str]] = Field(default=["duckduckgo", "reddit"])
    max_search_results: Optional[int] = Field(3, ge=1, le=10)
    
    # Reasoning parameters - FIXED: Use 'pattern' instead of 'regex'
    reasoning_mode: Optional[str] = Field("balanced", pattern="^(fast|balanced|deep)$")
    context_priority: Optional[str] = Field("recent", pattern="^(recent|relevant|mixed)$")
    response_format: Optional[str] = Field("conversational", pattern="^(conversational|structured|json)$")
    
class ExecutionRequest(BaseModel):
    """Enhanced request model for agent execution"""
    input: str = Field(..., min_length=1, max_length=10000)
    parameters: Optional[ExecutionParameters] = Field(default_factory=ExecutionParameters)
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ExecutionResponse(BaseModel):
    """Standardized response model"""
    success: bool
    response: str
    model: str
    agent_id: str
    execution_time_ms: float
    personality_applied: Dict[str, Any]
    context_length: int
    search_info: Dict[str, Any]
    reasoning_steps: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/{agent_id}", response_model=ExecutionResponse)
async def execute_agent_endpoint(
    agent_id: str,
    request_data: ExecutionRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute an AI agent with enhanced capabilities including internet search.
    
    Features:
    - Personality-aware responses
    - Multi-source internet search (DuckDuckGo, Reddit, Wikipedia, Hacker News)
    - Conversation memory
    - Flexible parameter control
    - Structured error handling
    """
    start_time = time.time()
    
    try:
        # Validate agent_id format
        if not agent_id or len(agent_id.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent ID cannot be empty"
            )
        
        # Convert request to service input format
        input_data = {
            "input": request_data.input,
            "parameters": jsonable_encoder(request_data.parameters.dict()) if request_data.parameters else {},
            "context": request_data.context or {},
            "metadata": request_data.metadata or {}
        }
        
        # Execute agent
        result = await execute_agent(agent_id, user["sub"], input_data, db)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000
        
        # Build response
        response = ExecutionResponse(
            success=result.get("success", True),
            response=result.get("response", ""),
            model=result.get("model", "unknown"),
            agent_id=agent_id,
            execution_time_ms=round(execution_time, 2),
            personality_applied=result.get("personality_applied", {}),
            context_length=result.get("context_length", 0),
            search_info=result.get("search_info", {"performed": False}),
            reasoning_steps=result.get("reasoning_steps"),
            sources=result.get("sources"),
            metadata={
                "user_id": user["sub"],
                "timestamp": time.time(),
                "request_id": f"{agent_id}_{int(time.time())}"
            },
            error=result.get("error")
        )
        
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error for agent {agent_id}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Agent not found: {str(ve)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution failed for agent {agent_id}: {str(e)}", exc_info=True)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Return structured error response
        return ExecutionResponse(
            success=False,
            response="I apologize, but I encountered an error processing your request.",
            model="unknown",
            agent_id=agent_id,
            execution_time_ms=round(execution_time, 2),
            personality_applied={},
            context_length=0,
            search_info={"performed": False, "error": str(e)},
            error=str(e),
            metadata={
                "user_id": user["sub"],
                "timestamp": time.time(),
                "error_type": type(e).__name__
            }
        )

# Backwards compatibility endpoint
@router.post("/{agent_id}/legacy")
async def execute_agent_legacy(
    agent_id: str,
    request_data: dict,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Legacy endpoint for backwards compatibility"""
    try:
        # Convert legacy format to new format
        execution_request = ExecutionRequest(
            input=request_data.get("input", ""),
            parameters=ExecutionParameters(**request_data.get("parameters", {}))
        )
        
        return await execute_agent_endpoint(agent_id, execution_request, user, db)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")
