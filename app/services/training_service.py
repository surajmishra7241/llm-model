import os
import asyncio
import aiofiles
import uuid
import logging
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
from pathlib import Path

from app.models.training_model import (
    TrainingJobResponse, 
    TrainingJobStatus,
    TrainingDataItem,
    TrainingDataType,
    TrainingProgress,
    FileUploadInfo
)
from app.utils.file_processing import (
    process_document,
    process_image,
    extract_text_from_url,
    chunk_text
)

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        # Create temporary directory for file uploads
        self.temp_dir = Path(tempfile.gettempdir()) / "training_uploads"  
        self.temp_dir.mkdir(exist_ok=True)
        
        # File storage with user isolation
        self.user_files = {}  # user_id -> {file_id: file_path}

    async def create_job(
        self,
        user_id: str,
        agent_id: str,
        data_urls: List[str] = None,
        training_data: List[TrainingDataItem] = None,
        text_data: List[str] = None,
        uploaded_files: List[FileUploadInfo] = None,
        config: Dict[str, Any] = None
    ) -> TrainingJobResponse:
        """Create a new enhanced training job"""
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Convert legacy data_urls to training_data format for backward compatibility
        enhanced_training_data = training_data or []
        if data_urls:
            for url in data_urls:
                enhanced_training_data.append(
                    TrainingDataItem(
                        type=TrainingDataType.URL,
                        content=url,
                        metadata={"source": "legacy_data_urls"}
                    )
                )
        
        # Add text data as training items
        if text_data:
            for i, text in enumerate(text_data):
                enhanced_training_data.append(
                    TrainingDataItem(
                        type=TrainingDataType.TEXT,
                        content=text,
                        metadata={"index": i, "source": "text_input"}
                    )
                )
        
        job = TrainingJobResponse(
            id=job_id,
            user_id=user_id,
            agent_id=agent_id,
            data_urls=data_urls or [],
            training_data=enhanced_training_data,
            text_data=text_data or [],
            uploaded_files=uploaded_files or [],
            status=TrainingJobStatus.PENDING,
            created_at=now,
            updated_at=now,
            progress=0,
            current_step="Initializing",
            total_steps=self._calculate_total_steps(enhanced_training_data),
            result=None,
            processed_items=0,
            total_items=len(enhanced_training_data)
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} with {len(enhanced_training_data)} training items")
        return job

    def _calculate_total_steps(self, training_data: List[TrainingDataItem]) -> int:
        """Calculate total processing steps based on data types"""
        steps = 1  # Initialization
        
        for item in training_data:
            if item.type == TrainingDataType.URL:
                steps += 2  # Download + Process
            elif item.type == TrainingDataType.FILE:
                steps += 2  # Read + Process
            elif item.type == TrainingDataType.IMAGE:
                steps += 2  # Read + OCR/Analysis
            else:  # TEXT
                steps += 1  # Process
        
        steps += 2  # Finalization + Model training
        return steps

    async def save_uploaded_file(
        self,
        user_id: str,
        file_id: str,
        filename: str,
        content: bytes
    ) -> str:
        """Save uploaded file to temporary storage"""
        try:
            # Create user-specific directory
            user_dir = self.temp_dir / user_id
            user_dir.mkdir(exist_ok=True)
            
            # Save file with unique name
            file_path = user_dir / f"{file_id}_{filename}"
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Track file for cleanup
            if user_id not in self.user_files:
                self.user_files[user_id] = {}
            self.user_files[user_id][file_id] = str(file_path)
            
            logger.info(f"Saved file {filename} for user {user_id} as {file_id}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise

    async def list_jobs(self, user_id: str, agent_id: str) -> List[TrainingJobResponse]:
        """List training jobs for a user and agent"""
        user_jobs = []
        for job in self.jobs.values():
            if job.user_id == user_id and job.agent_id == agent_id:
                user_jobs.append(job)
        
        # Sort by creation date, newest first
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        return user_jobs

    async def get_job(self, user_id: str, job_id: str) -> Optional[TrainingJobResponse]:
        """Get a specific job for a user"""
        job = self.jobs.get(job_id)
        if job and job.user_id == user_id:
            return job
        return None

    async def get_progress(self, user_id: str, job_id: str) -> Optional[TrainingProgress]:
        """Get training progress for a job"""
        job = await self.get_job(user_id, job_id)
        if not job:
            return None
        
        return TrainingProgress(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
            processed_items=job.processed_items,
            total_items=job.total_items,
            message=job.error_message if job.status == TrainingJobStatus.FAILED else None
        )

    async def cancel_job(self, user_id: str, job_id: str) -> bool:
        """Cancel a training job"""
        job = await self.get_job(user_id, job_id)
        if not job:
            return False
        
        if job.status in [TrainingJobStatus.PENDING, TrainingJobStatus.PROCESSING_FILES, 
                         TrainingJobStatus.EXTRACTING_CONTENT, TrainingJobStatus.RUNNING]:
            job.status = TrainingJobStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.updated_at = datetime.utcnow()
            logger.info(f"Cancelled job {job_id} for user {user_id}")
            return True
        
        return False

    async def run_training(self, job_id: str, user_id: str):
        """Run enhanced training process in background"""
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        if job.user_id != user_id:
            logger.error(f"User {user_id} not authorized for job {job_id}")
            return
        
        try:
            await self._execute_training_pipeline(job)
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {str(e)}")
            job.status = TrainingJobStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.utcnow()
        finally:
            # Cleanup uploaded files
            await self._cleanup_user_files(user_id)

    async def _cleanup_user_files(self, user_id: str):
        """Clean up uploaded files for a user"""
        try:
            if user_id in self.user_files:
                for file_id, file_path in self.user_files[user_id].items():
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Cleaned up file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up file {file_path}: {str(e)}")
                
                # Remove user directory if empty
                user_dir = self.temp_dir / user_id
                if user_dir.exists() and not any(user_dir.iterdir()):
                    user_dir.rmdir()
                
                # Clear from tracking
                del self.user_files[user_id]
                
        except Exception as e:
            logger.error(f"Error during cleanup for user {user_id}: {str(e)}")

    async def _execute_training_pipeline(self, job: TrainingJobResponse):
        """Execute the complete training pipeline"""
        logger.info(f"Starting training pipeline for job {job.id}")
        
        job.status = TrainingJobStatus.PROCESSING_FILES
        job.current_step = "Processing input data"
        job.updated_at = datetime.utcnow()
        
        processed_content = []
        current_step = 0
        
        # Process each training data item
        for i, item in enumerate(job.training_data):
            try:
                job.current_step = f"Processing item {i+1}/{len(job.training_data)}: {item.type.value}"
                job.processed_items = i
                job.progress = int((current_step / job.total_steps) * 100)
                job.updated_at = datetime.utcnow()
                
                content = await self._process_training_item(item)
                if content:
                    processed_content.extend(content)
                
                current_step += 2 if item.type != TrainingDataType.TEXT else 1
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}")
                # Continue with other items instead of failing completely
                continue
        
        # Update progress
        job.status = TrainingJobStatus.EXTRACTING_CONTENT
        job.current_step = "Extracting and chunking content"
        job.progress = 80
        job.updated_at = datetime.utcnow()
        
        # Chunk all processed content
        all_chunks = []
        for content in processed_content:
            chunks = chunk_text(content)
            all_chunks.extend(chunks)
        
        current_step += 1
        
        # Simulate model training
        job.status = TrainingJobStatus.RUNNING
        job.current_step = "Training model"
        job.progress = 90
        job.updated_at = datetime.utcnow()
        
        # Simulate training time based on content size
        training_time = min(len(all_chunks) * 0.1, 10)  # Max 10 seconds
        await asyncio.sleep(training_time)
        
        # Complete the job
        job.status = TrainingJobStatus.COMPLETED
        job.current_step = "Training completed"
        job.progress = 100
        job.processed_items = len(job.training_data)
        job.result = {
            "total_content_items": len(processed_content),
            "total_chunks": len(all_chunks),
            "average_chunk_size": sum(len(chunk) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0,
            "training_time_seconds": training_time,
            "data_types_processed": list(set(item.type.value for item in job.training_data)),
            "accuracy": 0.95,  # Simulated
            "loss": 0.15       # Simulated
        }
        job.updated_at = datetime.utcnow()
        
        logger.info(f"Training completed for job {job.id} with {len(all_chunks)} chunks")

    async def _process_training_item(self, item: TrainingDataItem) -> List[str]:
        """Process a single training data item"""
        try:
            if item.type == TrainingDataType.URL:
                return await self._process_url(item.content)
            elif item.type == TrainingDataType.FILE:
                return await self._process_file(item.content, item.metadata)
            elif item.type == TrainingDataType.IMAGE:
                return await self._process_image_file(item.content, item.metadata)
            elif item.type == TrainingDataType.TEXT:
                return [item.content]
            else:
                logger.warning(f"Unknown data type: {item.type}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing {item.type} item: {str(e)}")
            return []

    async def _process_url(self, url: str) -> List[str]:
        """Process content from URL"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                self.executor,
                extract_text_from_url,
                url
            )
            return [content] if content else []
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return []

    async def _process_file(self, file_path: str, metadata: Dict[str, Any]) -> List[str]:
        """Process uploaded file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []
            
            filename = metadata.get('filename', 'unknown')
            
            # Read file content
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Process document
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.executor,
                process_document,
                filename,
                content
            )
            
            return [text] if text else []
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

    async def _process_image_file(self, file_path: str, metadata: Dict[str, Any]) -> List[str]:
        """Process image file with OCR"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Image file not found: {file_path}")
                return []
            
            filename = metadata.get('filename', 'unknown')
            
            # Read image content
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Process image
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.executor,
                process_image,
                filename,
                content
            )
            
            return [text] if text else []
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            return []