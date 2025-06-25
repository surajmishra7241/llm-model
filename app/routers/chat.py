from fastapi import APIRouter, Depends, HTTPException
from app.models.response_schema import ChatRequest, ChatResponse
from app.services.llm_service import OllamaService
from app.dependencies import get_current_user

# Only one APIRouter definition, without a conflicting prefix
router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    chat_request: ChatRequest,
    llm_service: OllamaService = Depends(),
    user: dict = Depends(get_current_user)
):
    """Chat with an agent"""
    try:
        messages = [
            {"role": "system", "content": chat_request.system_prompt},
            {"role": "user", "content": chat_request.message}
        ]
        
        response = await llm_service.chat(
            messages=messages,
            model=chat_request.model,
            options=chat_request.options
        )
        
        return ChatResponse(
            message=response.get("message", {}).get("content", ""),
            model=chat_request.model,
            context=response.get("context", []),
            tokens_used=response.get("eval_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
