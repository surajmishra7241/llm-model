# utils/chroma_async.py
import chromadb
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial
from contextlib import asynccontextmanager
import time
import threading
from collections import defaultdict
from app.config import settings

logger = logging.getLogger(__name__)

class AsyncChromaClient:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, path: str, embedding_function: Optional[Callable] = None, max_workers: int = 4):
        if not hasattr(self, '_initialized'):
            self.path = path
            self.embedding_function = embedding_function
            self.executor = ThreadPoolExecutor(
                max_workers=max_workers, 
                thread_name_prefix="chroma",
            )
            self._local = threading.local()
            self.collections = defaultdict(dict)
            self._initialized = True
            self._closed = False
    
    async def initialize(self):
        """Initialize the client in a thread-safe way"""
        if not hasattr(self._local, 'client') or self._local.client is None:
            loop = asyncio.get_running_loop()
            self._local.client = await loop.run_in_executor(
                self.executor,
                partial(
                    chromadb.PersistentClient,
                    path=self.path,
                )
            )
            logger.info("AsyncChromaDB client initialized for thread %s", threading.current_thread().name)
    
    @asynccontextmanager
    async def get_collection(self, name: str, embedding_function=None, metadata=None) -> AsyncIterator[chromadb.Collection]:
        """Async context manager for collection access with connection pooling"""
        if self._closed:
            raise RuntimeError("Client is closed")
            
        await self.initialize()
        
        thread_id = threading.current_thread().ident
        collection_key = f"{name}_{thread_id}"
        
        if collection_key not in self.collections[thread_id]:
            loop = asyncio.get_running_loop()
            
            # Fix: Don't pass metadata parameter if it's None or empty
            collection_kwargs = {
                'name': name,
                'embedding_function': embedding_function or self.embedding_function,
            }
            
            # Only add metadata if it's not None and not empty
            if metadata:
                collection_kwargs['metadata'] = metadata
                
            collection = await loop.run_in_executor(
                self.executor,
                partial(
                    self._local.client.get_or_create_collection,
                    **collection_kwargs
                )
            )
            self.collections[thread_id][collection_key] = collection
        
        try:
            yield self.collections[thread_id][collection_key]
        except Exception as e:
            logger.error(f"Error in collection context for {name}: {str(e)}")
            raise
        finally:
            # Clean up if this is the last reference
            pass
    
    async def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        metadatas: List[Dict], 
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 100
    ):
        """Async document addition with improved batching and error handling"""
        if not documents:
            return
            
        async with self.get_collection(collection_name) as collection:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size] if metadatas else None
                batch_ids = ids[i:i + batch_size] if ids else None
                batch_embeds = embeddings[i:i + batch_size] if embeddings else None
                
                loop = asyncio.get_running_loop()
                
                try:
                    await loop.run_in_executor(
                        self.executor,
                        partial(
                            collection.add,
                            documents=batch_docs,
                            metadatas=batch_metas,
                            ids=batch_ids,
                            embeddings=batch_embeds
                        )
                    )
                    
                    # Small delay between batches to prevent overwhelming the system
                    if i + batch_size < len(documents):
                        await asyncio.sleep(0.05)
                        
                except Exception as e:
                    logger.error(f"Failed to add batch {i//batch_size + 1}: {str(e)}")
                    raise
    
    async def query_collection(
        self, 
        collection_name: str, 
        query_texts: Optional[List[str]] = None, 
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5, 
        where: Optional[Dict] = None, 
        where_document: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Async query with improved error handling and timeout"""
        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided")
            
        async with self.get_collection(collection_name) as collection:
            loop = asyncio.get_running_loop()
            
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        partial(
                            collection.query,
                            query_texts=query_texts,
                            query_embeddings=query_embeddings,
                            n_results=n_results,
                            where=where,
                            where_document=where_document,
                            include=include or ["documents", "distances", "metadatas"]
                        )
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Query timeout for collection {collection_name}")
                raise
            except Exception as e:
                logger.error(f"Query failed for collection {collection_name}: {str(e)}")
                raise
    
    async def close(self):
        """Cleanup resources"""
        if self._closed:
            return
            
        self._closed = True
        try:
            # Clear collections cache
            self.collections.clear()
            
            # Shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                logger.info("AsyncChromaDB client closed")
                
        except Exception as e:
            logger.error(f"Error closing AsyncChromaDB client: {str(e)}")
        finally:
            self._instance = None

# Global instance with lazy initialization
_async_chroma_client = None
_async_chroma_lock = asyncio.Lock()

async def get_async_chroma_client() -> AsyncChromaClient:
    global _async_chroma_client
    async with _async_chroma_lock:
        if _async_chroma_client is None or _async_chroma_client._closed:
            _async_chroma_client = AsyncChromaClient(
                path=settings.CHROMA_PATH,
                embedding_function=None,  # Will be set per request
                max_workers=settings.CHROMA_MAX_WORKERS
            )
        return _async_chroma_client