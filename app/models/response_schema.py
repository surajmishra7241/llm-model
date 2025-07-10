from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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



# response_schema.py
class RAGQueryRequest(BaseModel):
    query: str
    max_results: int = Field(5, gt=0, le=20)
    rewrite_query: bool = Field(True, description="Enable query rewriting")
    use_reranking: bool = Field(True, description="Enable cross-encoder re-ranking")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for retrieval")
    hybrid_search: bool = Field(True, description="Enable hybrid sparse+dense search")

class RAGResponse(BaseModel):
    answer: str
    documents: List[str]
    sources: List[str]
    context: List[str]
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information about the search process")
    search_method: Optional[str] = Field(None, description="Search method used")
    processed_query: Optional[str] = Field(None, description="Processed query after rewriting")

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str = "success"

class HybridSearchConfig(BaseModel):
    sparse_weight: float = Field(0.4, ge=0, le=1)
    dense_weight: float = Field(0.6, ge=0, le=1)
    enable_hybrid: bool = True