# config.py

from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from pydantic import Field, PostgresDsn, validator, RedisDsn, HttpUrl, conint
import logging
from urllib.parse import urlparse
import os

class Settings(BaseSettings):
    # App settings
    APP_VERSION: str = "1.0.0"
    APP_NAME: str = "AI Agent Platform"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "llm_agents"
    DATABASE_URL: Optional[PostgresDsn] = None
    


    

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if isinstance(v, str):
            return v
    
    # Fix: Don't add leading slash to database name
        db_name = values.get("POSTGRES_DB", "")
    
        return str(PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=db_name,  # Remove the leading slash
        ))


    
    # Redis
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 300
    
    # Ollama Configuration
    OLLAMA_URL: HttpUrl = "http://localhost:11434"
    DEFAULT_OLLAMA_MODEL: str = "deepseek-r1:1.5b"
    OLLAMA_TIMEOUT: conint(gt=0) = 60  # Increased timeout
    
    # Qdrant Configuration
    QDRANT_URL: HttpUrl = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_MAX_WORKERS: conint(gt=0) = 8
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_BATCH_SIZE: conint(gt=0) = 100
    QDRANT_TIMEOUT: conint(gt=0) = 30

    RATE_LIMITING_ENABLED: bool = True
    RATE_LIMIT_MAX_REQUESTS: int = 1000
    RATE_LIMIT_TIME_WINDOW: int = 60  # seconds

    HEALTH_CHECK_TIMEOUT: int = 5000
    PYTHON_API_TIMEOUT: int = 10000

    # PYTHON_API_URL=http://localhost:8000
    # PYTHON_VOICE_WS_URL=ws://localhost:8000/api/v1/voice/ws


    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_SPARSE_WEIGHT: float = Field(0.4, ge=0, le=1)
    HYBRID_DENSE_WEIGHT: float = Field(0.6, ge=0, le=1)
    
    # Query Processing
    QUERY_REWRITING_ENABLED: bool = True
    QUERY_EXPANSION_ENABLED: bool = True
    
    # Re-ranking
    RERANKING_ENABLED: bool = True
    RERANKING_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Metadata Filtering
    ENABLE_METADATA_FILTERING: bool = True
    DEFAULT_METADATA_FIELDS: List[str] = ["source", "author", "date", "document_type"]
    
    # Embedding Model Configuration
    # Use the same model as default for embeddings initially
    EMBEDDING_MODEL: str = "deepseek-r1:1.5b"  # Changed to use main model
    EXPECTED_EMBEDDING_DIMENSION: Optional[conint(gt=0)] = None
    EMBEDDING_FALLBACK_DIMENSION: conint(gt=0) = 768
    
    # Alternative embedding models to try (in order of preference)
    EMBEDDING_MODEL_ALTERNATIVES: List[str] = [
        "deepseek-r1:1.5b",
        "nomic-embed-text",
        "all-minilm",
        "mxbai-embed-large"
    ]

    # Voice settings
    ENABLE_WHISPER: bool = True
    WHISPER_MODEL: str = "base"  # or "small", "medium", "large"
    ENABLE_TTS: bool = True
    TTS_ENGINE: str = "coqui"  # or "pyttsx3"
    TTS_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    TTS_VOICE: str = "male"
    TTS_MODEL_ALTERNATIVES: List[str] = [
        "tts_models/en/ljspeech/tacotron2-DDC",    # Most reliable single-speaker
        "tts_models/en/ljspeech/fast_pitch",       # Fast single-speaker
        "tts_models/en/ljspeech/glow-tts",         # Good quality single-speaker
        "tts_models/en/ljspeech/neural_hmm",       # Alternative single-speaker
        "tts_models/en/vctk/vits",                 # Multi-speaker (fallback)
    ]

    TTS_MAX_CHUNK_SIZE: int = 400  # Maximum characters per TTS chunk
    TTS_ENABLE_CHUNKING: bool = True  # Enable text chunking for long responses
    TTS_CHUNK_PAUSE_DURATION: float = 0.3  # Pause between chunks in seconds
    
    # Document Processing
    DEFAULT_CHUNK_SIZE: conint(gt=0) = 1000
    DEFAULT_CHUNK_OVERLAP: conint(ge=0) = 200
    MAX_DOCUMENT_SIZE_MB: conint(gt=0) = 50
    SUPPORTED_FILE_TYPES: List[str] = [".pdf", ".txt", ".docx", ".pptx", ".xlsx"]
    
    # RAG Settings
    RAG_DEFAULT_MAX_RESULTS: conint(gt=0) = 5
    RAG_DEFAULT_MIN_SCORE: float = Field(ge=0, le=1, default=0.3)
    RAG_QUERY_TIMEOUT: conint(gt=0) = 30
    
    # Auth
    JWT_SECRET_KEY: str = "your-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 11440  # 24 hours



    ENABLE_INTERNET_SEARCH: bool = True
    SEARCH_SOURCES: List[str] = ["duckduckgo", "reddit", "hackernews", "wikipedia"]
    MAX_SEARCH_RESULTS_PER_SOURCE: int = 5
    SEARCH_TIMEOUT_SECONDS: int = 15

    # Google Search API (optional - for paid Google Custom Search)
    GOOGLE_SEARCH_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = None

    # Brave Search API (optional - for Brave Search API)
    BRAVE_SEARCH_API_KEY: Optional[str] = None

    # Rate limiting for search APIs
    SEARCH_RATE_LIMIT_PER_MINUTE: int = 60
    SEARCH_CACHE_TTL_SECONDS: int = 3600  # 1 hour



# WebSocket Configuration
    WEBSOCKET_MAX_CONNECTIONS: int = 100
    WEBSOCKET_TIMEOUT: int = 300
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: conint(gt=0, le=65535) = 8001
    
    # Logging
    LOG_LEVEL: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", default="INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Embedding fallback strategy
    USE_DETERMINISTIC_FALLBACK: bool = True
    FALLBACK_EMBEDDING_STRATEGY: str = Field(
        pattern="^(deterministic|hash|features|skip)$", 
        default="deterministic"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

# Create settings instance
settings = Settings()

def validate_config():
    """Validate and log configuration"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate URLs
        parsed_ollama = urlparse(str(settings.OLLAMA_URL))
        if not all([parsed_ollama.scheme, parsed_ollama.netloc]):
            raise ValueError("Invalid OLLAMA_URL format")
            
        parsed_qdrant = urlparse(str(settings.QDRANT_URL))
        if not all([parsed_qdrant.scheme, parsed_qdrant.netloc]):
            raise ValueError("Invalid QDRANT_URL format")
        
        # Log important config
        logger.info("=== Application Configuration ===")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug Mode: {settings.DEBUG}")
        logger.info("")
        logger.info("=== Ollama Configuration ===")
        logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
        logger.info(f"Default Model: {settings.DEFAULT_OLLAMA_MODEL}")
        logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"Timeout: {settings.OLLAMA_TIMEOUT}s")
        logger.info(f"Alternative Models: {settings.EMBEDDING_MODEL_ALTERNATIVES}")
        logger.info("")
        logger.info("=== Vector Database Configuration ===")
        logger.info(f"Qdrant URL: {settings.QDRANT_URL}")
        logger.info(f"Collection Name: {settings.QDRANT_COLLECTION_NAME}")
        logger.info(f"Expected Dimension: {settings.EXPECTED_EMBEDDING_DIMENSION or 'Auto-detect'}")
        logger.info(f"Fallback Dimension: {settings.EMBEDDING_FALLBACK_DIMENSION}")
        logger.info("")
        logger.info("=== Document Processing ===")
        logger.info(f"Chunk Size: {settings.DEFAULT_CHUNK_SIZE}")
        logger.info(f"Chunk Overlap: {settings.DEFAULT_CHUNK_OVERLAP}")
        logger.info(f"Max File Size: {settings.MAX_DOCUMENT_SIZE_MB}MB")
        logger.info(f"Supported Types: {settings.SUPPORTED_FILE_TYPES}")
        logger.info("")
        logger.info("=== RAG Configuration ===")
        logger.info(f"Max Results: {settings.RAG_DEFAULT_MAX_RESULTS}")
        logger.info(f"Min Score Threshold: {settings.RAG_DEFAULT_MIN_SCORE}")
        logger.info(f"Query Timeout: {settings.RAG_QUERY_TIMEOUT}s")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise

# Validate on import (only in production-like environments)
if "pytest" not in os.sys.modules and __name__ != "__main__":
    try:
        validate_config()
    except Exception as e:
        print(f"Warning: Configuration validation failed: {e}")
        # Don't raise in case this is being imported during setup