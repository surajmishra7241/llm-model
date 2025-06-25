import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import asynccontextmanager
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings # Ensure this import path is correct based on your project structure

logger = logging.getLogger(__name__)

class QdrantConnectionError(Exception):
    """Custom exception for Qdrant connection issues"""
    pass

class QdrantOperationError(Exception):
    """Custom exception for Qdrant operation failures"""
    pass

@dataclass
class CollectionInfo:
    name: str
    status: str
    vector_size: int
    distance: str
    points_count: int

class AsyncQdrantClient:
    """Thread-safe async wrapper for Qdrant client with improved error handling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, url: str = None, api_key: str = None, max_workers: int = None):
        if self._initialized:
            return
            
        self.url = url or str(settings.QDRANT_URL)
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.max_workers = max_workers or settings.QDRANT_MAX_WORKERS
        self.timeout = settings.QDRANT_TIMEOUT
        self.batch_size = settings.QDRANT_BATCH_SIZE
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers, 
            thread_name_prefix="qdrant_worker"
        )
        self._local = threading.local()
        self._initialized = True
        self._closed = False
        self._connection_verified = False
        
        logger.info(f"AsyncQdrantClient initialized with URL: {self.url}")
    
    def _get_client(self) -> QdrantClient:
        """Get thread-local client instance"""
        if not hasattr(self._local, 'client') or self._local.client is None:
            try:
                self._local.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    prefer_grpc=False, # Use HTTP API
                    timeout=self.timeout,
                    grpc_options={ # These are ignored if prefer_grpc is False, but kept for completeness
                        "grpc.keepalive_time_ms": 30000,
                        "grpc.max_receive_message_length": 100 * 1024 * 1024  # 100MB
                    }
                )
                logger.debug(f"Created new Qdrant client for thread {threading.current_thread().name}")
            except Exception as e:
                logger.error(f"Failed to create Qdrant client: {str(e)}", exc_info=True)
                raise QdrantConnectionError(f"Cannot connect to Qdrant at {self.url}: {str(e)}")
        
        return self._local.client
    
    async def _verify_connection(self) -> bool:
        """Verify Qdrant connection and service availability"""
        if self._connection_verified:
            return True
            
        try:
            loop = asyncio.get_running_loop()
            
            def _check_connection():
                client = self._get_client()
                try:
                    # First try to get collections as basic health check
                    collections = client.get_collections()
                    logger.info(f"Qdrant connection verified - {len(collections.collections)} collections found")
                    return True
                except Exception as e:
                    logger.error(f"Qdrant connection check failed: {str(e)}")
                    raise QdrantConnectionError(f"Cannot verify connection to Qdrant: {str(e)}")
            
            result = await loop.run_in_executor(self.executor, _check_connection)
            self._connection_verified = result
            logger.info("Qdrant connection verified successfully")
            return result
            
        except Exception as e:
            logger.error(f"Qdrant connection verification failed: {str(e)}", exc_info=True)
            raise QdrantConnectionError(f"Cannot verify connection to Qdrant: {str(e)}")
    
    async def initialize(self):
        """Initialize and verify the client connection"""
        if self._closed:
            raise QdrantConnectionError("Client has been closed")
            
        await self._verify_connection()
        logger.info("AsyncQdrant client initialized and verified")
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        recreate_if_exists: bool = False
    ) -> bool:
        """Create a new collection with specified parameters"""
        await self.initialize()
        
        try:
            loop = asyncio.get_running_loop()
            
            def _create_collection():
                client = self._get_client()
                
                # Check if collection exists
                try:
                    collections = client.get_collections()
                    existing_names = [col.name for col in collections.collections]
                    
                    if collection_name in existing_names:
                        if recreate_if_exists:
                            logger.info(f"Recreating existing collection: {collection_name}")
                            client.delete_collection(collection_name)
                            # Give a small moment for deletion to propagate if recreating
                            time.sleep(0.1) 
                        else:
                            logger.info(f"Collection {collection_name} already exists")
                            return True
                except Exception as e:
                    logger.warning(f"Error checking existing collections: {str(e)}")
                    # If checking collections fails, and we are not forcing recreate, raise
                    if not recreate_if_exists:
                        raise QdrantOperationError(f"Could not verify collection existence: {str(e)}")
            
                # Create collection with proper configuration
                distance_enum = Distance[distance.upper()] if hasattr(Distance, distance.upper()) else Distance.COSINE
                
                result = client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance_enum
                    )
                )
                
                logger.info(f"Created collection {collection_name} with {vector_size}D vectors")
                return True
            
            return await loop.run_in_executor(self.executor, _create_collection)
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}", exc_info=True)
            raise QdrantOperationError(f"Collection creation failed: {str(e)}")
    
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        await self.initialize()
        
        try:
            loop = asyncio.get_running_loop()
            
            def _check_collection():
                client = self._get_client()
                try:
                    collections = client.get_collections()
                    return collection_name in [col.name for col in collections.collections]
                except Exception as e:
                    logger.warning(f"Error checking collection existence: {str(e)}")
                    return False
            
            return await loop.run_in_executor(self.executor, _check_collection)
            
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}", exc_info=True)
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get collection information"""
        await self.initialize()
        
        try:
            loop = asyncio.get_running_loop()
            
            def _get_info():
                client = self._get_client()
                try:
                    info = client.get_collection(collection_name)
                    return CollectionInfo(
                        name=collection_name,
                        status=info.status.value if hasattr(info.status, 'value') else str(info.status),
                        vector_size=info.config.params.vectors.size,
                        distance=info.config.params.vectors.distance.value,
                        points_count=info.points_count
                    )
                except Exception as e:
                    logger.error(f"Error getting collection info: {str(e)}")
                    return None
            
            return await loop.run_in_executor(self.executor, _get_info)
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}", exc_info=True)
            return None
    
    async def upsert(
        self,
        collection_name: str,
        points: List[PointStruct],
        batch_size: int = None
    ) -> bool:
        """Upsert points into collection with batching"""
        await self.initialize()
        
        if not points:
            logger.warning("No points to upsert")
            return True
        
        batch_size = batch_size or self.batch_size
        
        try:
            loop = asyncio.get_running_loop()
            
            def _upsert_batch(batch_points):
                client = self._get_client()
                
                # Validate points before upsert
                for point in batch_points:
                    if not isinstance(point.vector, list) or len(point.vector) == 0:
                        raise ValueError(f"Invalid vector for point {point.id}")
                    if not isinstance(point.payload, dict):
                        raise ValueError(f"Invalid payload for point {point.id}")
                
                result = client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True
                )
                return result
            
            # Process in batches
            total_points = len(points)
            logger.info(f"Upserting {total_points} points in batches of {batch_size}")
            
            success_count = 0
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_points + batch_size - 1) // batch_size
                
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} points)")
                
                try:
                    result = await loop.run_in_executor(self.executor, _upsert_batch, batch)
                    if result and result.status == models.UpdateStatus.COMPLETED:
                        success_count += len(batch)
                    else:
                        logger.error(f"Batch {batch_num} upsert failed with status: {result.status if result else 'N/A'}")
                        raise QdrantOperationError(f"Batch {batch_num} upsert failed")
                    
                    # Small delay between batches to avoid overwhelming Qdrant
                    if batch_num < total_batches:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Failed to upsert batch {batch_num}: {str(e)}")
                    raise
            
            logger.info(f"Successfully upserted {success_count}/{total_points} points to {collection_name}")
            return success_count == total_points
            
        except Exception as e:
            logger.error(f"Failed to upsert points: {str(e)}", exc_info=True)
            raise QdrantOperationError(f"Upsert operation failed: {str(e)}")
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_filter: Optional[Filter] = None,
        limit: int = 5,
        with_vectors: bool = False,
        with_payload: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[models.ScoredPoint]:
        """Search collection with query vector"""
        await self.initialize()
        
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
        
        try:
            loop = asyncio.get_running_loop()
            
            def _search():
                client = self._get_client()
                
                results = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=max(1, min(limit, 1000)),
                    with_vectors=with_vectors,
                    with_payload=with_payload,
                    score_threshold=score_threshold
                )
                return results
            
            results = await loop.run_in_executor(self.executor, _search)
            
            logger.debug(f"Search returned {len(results)} results for collection {collection_name}")
            return results
            
        except ValueError as ve:
            logger.error(f"Validation error in search: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise QdrantOperationError(f"Search operation failed: {str(e)}")
    
    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Optional[Filter] = None,
        limit: int = 10,
        offset: Optional[str] = None,
        with_vectors: bool = False,
        with_payload: bool = True
    ) -> Tuple[List[models.Record], Optional[str]]:
        """Scroll through collection records"""
        await self.initialize()
        
        try:
            loop = asyncio.get_running_loop()
            
            def _scroll():
                client = self._get_client()
                records, next_page_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=max(1, min(limit, 1000)),
                    offset=offset,
                    with_vectors=with_vectors,
                    with_payload=with_payload
                )
                return records, next_page_offset
            
            records, next_offset = await loop.run_in_executor(self.executor, _scroll)
            logger.debug(f"Scroll returned {len(records)} records from {collection_name}")
            return records, next_offset
            
        except Exception as e:
            logger.error(f"Scroll failed: {str(e)}", exc_info=True)
            raise QdrantOperationError(f"Scroll operation failed: {str(e)}")
    
    async def delete(
        self,
        collection_name: str,
        points_selector: models.PointsSelector
    ) -> bool:
        """Delete points from collection"""
        await self.initialize()
        
        try:
            loop = asyncio.get_running_loop()
            
            def _delete():
                client = self._get_client()
                result = client.delete(
                    collection_name=collection_name,
                    points_selector=points_selector,
                    wait=True
                )
                return result
            
            result = await loop.run_in_executor(self.executor, _delete)
            logger.info(f"Delete operation completed for collection {collection_name}")
            return bool(result and result.status == models.UpdateStatus.COMPLETED)
            
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}", exc_info=True)
            raise QdrantOperationError(f"Delete operation failed: {str(e)}")
    
    async def count_points(
        self,
        collection_name: str,
        count_filter: Optional[Filter] = None
    ) -> int:
        """Count points in collection"""
        await self.initialize()
        
        try:
            loop = asyncio.get_running_loop()
            
            def _count():
                client = self._get_client()
                result = client.count(
                    collection_name=collection_name,
                    count_filter=count_filter,
                    exact=True
                )
                return result.count
            
            count = await loop.run_in_executor(self.executor, _count)
            return count
            
        except Exception as e:
            logger.error(f"Count failed: {str(e)}", exc_info=True)
            raise QdrantOperationError(f"Count operation failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            await self.initialize()
            
            loop = asyncio.get_running_loop()
            
            def _health_check():
                client = self._get_client()
                start_time = time.time()
                
                try:
                    collections = client.get_collections()
                    
                    end_time = time.time()
                    
                    return {
                        "status": "healthy",
                        "url": self.url,
                        "response_time_ms": round((end_time - start_time) * 1000, 2),
                        "collections_count": len(collections.collections),
                        "timestamp": time.time()
                    }
                except Exception as e:
                    return {
                        "status": "unhealthy",
                        "url": self.url,
                        "error": str(e),
                        "timestamp": time.time()
                    }
            
            return await loop.run_in_executor(self.executor, _health_check)
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": self.url,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def close(self):
        """Cleanup resources"""
        if self._closed:
            return
            
        self._closed = True
        
        try:
            # Shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True, timeout=30)
                logger.info("AsyncQdrant executor shutdown completed")
                
            if hasattr(self._local, 'client') and self._local.client is not None:
                try:
                    self._local.client.close()
                    logger.info("Main thread Qdrant client closed.")
                except Exception as e:
                    logger.warning(f"Error closing main thread client: {str(e)}")
        except Exception as e:
            logger.error(f"Error closing AsyncQdrant client: {str(e)}", exc_info=True)
        finally:
            # Reset the singleton instance to allow re-initialization if needed
            AsyncQdrantClient._instance = None 
            logger.info("AsyncQdrant client closed")

# Global instance management
_async_qdrant_client: Optional[AsyncQdrantClient] = None
_async_qdrant_lock = asyncio.Lock()

async def get_async_qdrant_client() -> AsyncQdrantClient:
    """Get or create an async Qdrant client instance"""
    global _async_qdrant_client
    
    async with _async_qdrant_lock:
        if _async_qdrant_client is None or _async_qdrant_client._closed:
            _async_qdrant_client = AsyncQdrantClient()
            await _async_qdrant_client.initialize()
        return _async_qdrant_client

@asynccontextmanager
async def qdrant_client_context():
    """Context manager for Qdrant client"""
    client = await get_async_qdrant_client()
    try:
        yield client
    finally:
        # In a global singleton scenario, we generally don't close the client here
        # as it's meant to be reused across requests.
        # Closing should be handled at application shutdown.
        pass

# Utility functions
def create_user_filter(user_id: str) -> Filter:
    """Create a filter for user documents"""
    return Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )

def create_document_filter(user_id: str, doc_id: str) -> Filter:
    """Create a filter for specific document"""
    return Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            ),
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id)
            )
        ]
    )

def validate_vector(vector: List[float], expected_dim: int) -> bool:
    """Validate vector dimensions and values"""
    if not isinstance(vector, list):
        return False
    if len(vector) != expected_dim:
        return False
    if not all(isinstance(x, (int, float)) for x in vector):
        return False
    return True