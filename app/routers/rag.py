from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from typing import List, Optional
from app.models.response_schema import RAGResponse, DocumentResponse, RAGQueryRequest
from app.services.rag_service import RAGService
from app.dependencies import get_current_user, get_db
from sqlalchemy.ext.asyncio import AsyncSession
import logging

router = APIRouter(prefix="/rag", tags=["RAG Operations"])
logger = logging.getLogger(__name__)

def extract_user_id(user: dict) -> str:
    """Extract user ID from JWT payload or user dict"""
    if isinstance(user, dict):
        # JWT tokens typically use 'sub' (subject) for user ID
        user_id = (user.get('sub') or 
                  user.get('id') or 
                  user.get('user_id') or 
                  user.get('email'))
        if user_id:
            return str(user_id)
    
    raise ValueError(f"Could not extract user ID from user object: {user}")

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a document for RAG processing"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )

    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt', '.md')):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF, TXT, and MD files are supported"
            )

        contents = await file.read()
        
        # Extract user ID properly
        user_id = extract_user_id(user)
        
        doc_id = await rag_service.ingest_document(
            db=db,
            user_id=user_id,
            filename=file.filename,
            content=contents
        )

        return DocumentResponse(
            document_id=str(doc_id),
            filename=file.filename,
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document"
        )

@router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: RAGQueryRequest,  # Changed from individual parameters to request model
    rag_service: RAGService = Depends(),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Query documents using RAG"""
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 3 characters"
        )

    try:
        user_id = extract_user_id(user)
        
        results = await rag_service.query(
            db=db,
            user_id=user_id,
            query=request.query,
            max_results=request.max_results
        )
        
        return RAGResponse(
            answer=results.get("answer", "No answer found"),
            documents=results.get("documents", []),
            context=results.get("context", []),
            sources=results.get("sources", [])
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    page: int = 1,
    per_page: int = 10,
    rag_service: RAGService = Depends(),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all documents for the user with pagination"""
    try:
        if page < 1 or per_page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page and per_page must be positive integers"
            )

        user_id = extract_user_id(user)

        return await rag_service.list_documents(
            db=db,
            user_id=user_id,
            page=page,
            per_page=per_page
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific document"""
    try:
        # Clean the document ID by removing any quotes or special characters
        document_id = document_id.strip('"\'')
        user_id = extract_user_id(user)
        
        logger.info(f"Attempting to delete document {document_id} for user {user_id}")
        
        success = await rag_service.delete_document(
            db=db,
            user_id=user_id,
            document_id=document_id
        )
        
        if not success:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or not owned by user"
            )
            
        logger.info(f"Successfully deleted document {document_id}")
        return {"status": "success", "deleted_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed for {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )