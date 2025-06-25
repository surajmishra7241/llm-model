from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from app.models.training_model import (
    TrainingJobResponse, 
    TrainingJobCreate, 
    TrainingProgress,
    TrainingDataItem,
    TrainingDataType,
    is_supported_file,
    get_file_type,
    FileUploadInfo
)
from app.services.training_service import TrainingService
from app.dependencies import get_current_user
import logging
import uuid
import json
import os
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

# Global instance to maintain state across requests
_training_service_instance = TrainingService()

def get_training_service() -> TrainingService:
    """Get training service singleton instance"""
    return _training_service_instance

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

@router.post("", response_model=TrainingJobResponse)
async def create_training_job(
    training_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """Create a new training job with enhanced data support"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Creating training job for user_id: {user_id}, agent_id: {training_data.agent_id}")
        
        # Validate that at least some data is provided
        has_data = any([
            training_data.data_urls,
            training_data.training_data,
            training_data.text_data
        ])
        
        if not has_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one data source must be provided (data_urls, training_data, or text_data)"
            )
            
        job = await training_service.create_job(
            user_id=user_id,
            agent_id=training_data.agent_id,
            data_urls=training_data.data_urls or [],
            training_data=training_data.training_data or [],
            text_data=training_data.text_data or [],
            config=training_data.config or {}
        )
        
        background_tasks.add_task(
            training_service.run_training,
            job_id=job.id,
            user_id=user_id
        )
        
        logger.info(f"Created enhanced job: {job.id} for user: {user_id}, agent: {training_data.agent_id}")
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating training job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/upload", response_model=TrainingJobResponse)
async def create_training_job_with_files(
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    agent_id: str = Form(...),
    text_data: Optional[str] = Form(None),  # JSON string of text array
    config: Optional[str] = Form("{}"),     # JSON string of config
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """Create a training job with file uploads"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Creating training job with files for user_id: {user_id}, agent_id: {agent_id}")
        
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file must be uploaded"
            )
        
        uploaded_files = []
        training_data_items = []
        
        for file in files:
            if not file.filename:
                continue
                
            if not is_supported_file(file.filename):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File type not supported: {file.filename}"
                )
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            file_content = await file.read()
            
            # Store file information
            file_info = FileUploadInfo(
                filename=file.filename,
                content_type=file.content_type or "application/octet-stream",
                size=len(file_content),
                file_id=file_id
            )
            uploaded_files.append(file_info)
            
            # Save file temporarily for processing
            temp_path = await training_service.save_uploaded_file(
                user_id, file_id, file.filename, file_content
            )
            
            # Create training data item
            data_item = TrainingDataItem(
                type=get_file_type(file.filename),
                content=temp_path,
                metadata={
                    "filename": file.filename,
                    "file_id": file_id,
                    "content_type": file.content_type,
                    "size": len(file_content)
                }
            )
            training_data_items.append(data_item)
        
        # Parse text data if provided
        parsed_text_data = []
        if text_data:
            try:
                parsed_text_data = json.loads(text_data)
                if not isinstance(parsed_text_data, list):
                    raise ValueError("text_data must be a JSON array")
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON format for text_data"
                )
        
        # Parse config
        parsed_config = {}
        if config:
            try:
                parsed_config = json.loads(config)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON format for config"
                )
        
        job = await training_service.create_job(
            user_id=user_id,
            agent_id=agent_id,
            data_urls=[],
            training_data=training_data_items,
            text_data=parsed_text_data,
            uploaded_files=uploaded_files,
            config=parsed_config
        )
        
        background_tasks.add_task(
            training_service.run_training,
            job_id=job.id,
            user_id=user_id
        )
        
        logger.info(f"Created file-based job: {job.id} with {len(uploaded_files)} files")
        return job
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating training job with files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    agent_id: str,
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """List training jobs for an agent"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Listing jobs for user_id: {user_id}, agent_id: {agent_id}")
        
        jobs = await training_service.list_jobs(user_id, agent_id)
        logger.info(f"Found {len(jobs)} jobs for user: {user_id}, agent: {agent_id}")
        
        return jobs
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing training jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@router.get("/{job_id}/progress", response_model=TrainingProgress)
async def get_training_progress(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """Get real-time training progress"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Getting progress for job: {job_id}, user: {user_id}")
        
        progress = await training_service.get_progress(user_id, job_id)
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or access denied"
            )
        
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{job_id}")
async def cancel_training_job(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """Cancel a training job"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Cancelling job: {job_id} for user: {user_id}")
        
        success = await training_service.cancel_job(user_id, job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or cannot be cancelled"
            )
        
        return {"message": "Job cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/debug/all-jobs")
async def debug_all_jobs(
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """Debug endpoint to see all jobs in memory"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Debug: Getting all jobs for user: {user_id}")
        
        # Access the jobs dictionary directly for debugging
        all_jobs = []
        for job_id, job in training_service.jobs.items():
            job_info = {
                "job_id": job_id,
                "user_id": job.user_id,
                "agent_id": job.agent_id,
                "status": job.status,
                "created_at": job.created_at,
                "data_sources": {
                    "urls": len(job.data_urls),
                    "training_data": len(job.training_data),
                    "text_data": len(job.text_data),
                    "uploaded_files": len(job.uploaded_files)
                }
            }
            all_jobs.append(job_info)
        
        # Filter for current user
        user_jobs = [job for job in all_jobs if job["user_id"] == user_id]
        
        return {
            "current_user_id": user_id,
            "total_jobs_in_system": len(all_jobs),
            "jobs_for_current_user": len(user_jobs),
            "all_jobs": all_jobs,
            "user_jobs": user_jobs
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
    user: dict = Depends(get_current_user)
):
    """Get training job details"""
    try:
        user_id = extract_user_id(user)
        logger.info(f"Getting job: {job_id} for user: {user_id}")
        
        job = await training_service.get_job(user_id, job_id)
        if not job:
            logger.warning(f"Job not found: {job_id} for user: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Job not found"
            )
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )