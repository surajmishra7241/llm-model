from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import mimetypes

class TrainingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING_FILES = "processing_files"
    EXTRACTING_CONTENT = "extracting_content"

class TrainingDataType(str, Enum):
    URL = "url"
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"

class TrainingDataItem(BaseModel):
    type: TrainingDataType
    content: str  # URL, text content, or file path
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('content')
    def validate_content(cls, v, values):
        data_type = values.get('type')
        if data_type == TrainingDataType.URL:
            # Basic URL validation
            if not v.startswith(('http://', 'https://')):
                raise ValueError('URL must start with http:// or https://')
        elif data_type == TrainingDataType.TEXT:
            if len(v.strip()) < 10:
                raise ValueError('Text content must be at least 10 characters')
        return v

class TrainingJobCreate(BaseModel):
    agent_id: str
    # Backward compatibility - keep data_urls for existing API
    data_urls: Optional[List[str]] = None
    # New enhanced data structure
    training_data: Optional[List[TrainingDataItem]] = None
    # Direct text input
    text_data: Optional[List[str]] = None
    # Training configuration
    config: Optional[Dict[str, Any]] = {}
    
    @validator('training_data', 'data_urls', 'text_data')
    def validate_data_sources(cls, v, values):
        # Ensure at least one data source is provided
        data_urls = values.get('data_urls')
        text_data = values.get('text_data')
        
        if not any([v, data_urls, text_data]):
            raise ValueError('At least one data source must be provided')
        return v

class FileUploadInfo(BaseModel):
    filename: str
    content_type: str
    size: int
    file_id: str

class TrainingJobResponse(BaseModel):
    id: str
    user_id: str
    agent_id: str
    data_urls: List[str] = []
    training_data: List[TrainingDataItem] = []
    text_data: List[str] = []
    uploaded_files: List[FileUploadInfo] = []
    status: TrainingJobStatus
    created_at: datetime
    updated_at: datetime
    progress: int
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processed_items: int = 0
    total_items: int = 0

class TrainingProgress(BaseModel):
    job_id: str
    status: TrainingJobStatus
    progress: int
    current_step: Optional[str] = None
    processed_items: int = 0
    total_items: int = 0
    message: Optional[str] = None

# Supported file types
SUPPORTED_EXTENSIONS = {
    # Text files
    '.txt', '.md', '.rtf', '.csv', '.json', '.xml', '.html', '.htm',
    # Documents
    '.pdf', '.doc', '.docx', '.odt', '.pages',
    # Images
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg',
    # Audio (for future transcription)
    '.mp3', '.wav', '.m4a', '.flac', '.ogg',
    # Video (for future transcription)
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'
}

def get_file_type(filename: str) -> TrainingDataType:
    """Determine the training data type based on file extension"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    ext = f'.{ext}'
    
    if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}:
        return TrainingDataType.IMAGE
    else:
        return TrainingDataType.FILE

def is_supported_file(filename: str) -> bool:
    """Check if file extension is supported"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    return f'.{ext}' in SUPPORTED_EXTENSIONS