from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    message: str
    model: str
    context: List[Any]
    tokens_used: int

class RAGResponse(BaseModel):
    answer: str
    documents: List[str]
    context: List[Any]
    sources: List[str] = []

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str = "success" 

class RAGQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5