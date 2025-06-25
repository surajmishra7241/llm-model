# service/rag_service.py
import logging
import os
import asyncio
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.file_processing import process_document
from app.services.llm_service import OllamaService
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.qdrant_async import get_async_qdrant_client
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)

def chunk_text_with_overlap(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into chunks with overlap to maintain context.
    
    Args:
        text: The input text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separator: Primary separator to use for splitting (paragraphs by default)
    
    Returns:
        List of text chunks with overlap
    """
    if not text or not text.strip():
        return []
    
    # If text is smaller than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    
    # First, try to split by separator (paragraphs)
    paragraphs = text.split(separator)
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk_size
        if len(current_chunk) + len(paragraph) + len(separator) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # Take last chunk_overlap characters from current chunk
                    overlap_text = current_chunk[-chunk_overlap:].strip()
                    # Find a good breaking point (sentence or word boundary)
                    sentences = overlap_text.split('. ')
                    if len(sentences) > 1:
                        overlap_text = '. '.join(sentences[1:])
                    current_chunk = overlap_text + separator + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Single paragraph is larger than chunk_size, need to split it
                if len(paragraph) > chunk_size:
                    # Split large paragraph by sentences
                    sentences = paragraph.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if not sentence.endswith('.') and sentence != sentences[-1]:
                            sentence += '.'
                            
                        if len(temp_chunk) + len(sentence) + 1 > chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                
                                # Add overlap
                                if chunk_overlap > 0 and len(temp_chunk) > chunk_overlap:
                                    overlap_text = temp_chunk[-chunk_overlap:].strip()
                                    temp_chunk = overlap_text + " " + sentence
                                else:
                                    temp_chunk = sentence
                            else:
                                # Single sentence is too large, split by words
                                words = sentence.split()
                                word_chunk = ""
                                
                                for word in words:
                                    if len(word_chunk) + len(word) + 1 > chunk_size:
                                        if word_chunk:
                                            chunks.append(word_chunk.strip())
                                            
                                            # Add overlap
                                            if chunk_overlap > 0:
                                                overlap_words = word_chunk.split()[-chunk_overlap//10:]  # Rough estimate
                                                word_chunk = " ".join(overlap_words) + " " + word
                                            else:
                                                word_chunk = word
                                        else:
                                            # Single word is too large, just add it
                                            chunks.append(word)
                                            word_chunk = ""
                                    else:
                                        word_chunk = word_chunk + " " + word if word_chunk else word
                                
                                if word_chunk:
                                    temp_chunk = word_chunk
                                else:
                                    temp_chunk = ""
                        else:
                            temp_chunk = temp_chunk + " " + sentence if temp_chunk else sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
        else:
            current_chunk = current_chunk + separator + paragraph if current_chunk else paragraph
    
    # Add the last chunk
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Clean up chunks - remove empty ones and duplicates
    cleaned_chunks = []
    seen_chunks = set()
    
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and chunk not in seen_chunks and len(chunk) > 50:  # Minimum chunk size
            cleaned_chunks.append(chunk)
            seen_chunks.add(chunk)
    
    logger.info(f"Split text into {len(cleaned_chunks)} chunks with overlap")
    return cleaned_chunks

class OllamaEmbeddingFunction:
    def __init__(self, ollama_service: OllamaService):
        self.ollama = ollama_service
        self._embedding_dim = None
    
    async def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model"""
        if self._embedding_dim is None:
            try:
                # Test with a small text to get the dimension
                test_embedding = await self.ollama.create_embedding("test")
                self._embedding_dim = len(test_embedding) if test_embedding else 384
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to detect embedding dimension: {str(e)}")
                self._embedding_dim = 384  # Default fallback
        return self._embedding_dim
    
    async def generate_embeddings(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        embedding_dim = await self._get_embedding_dimension()
        
        for text in input:
            try:
                logger.debug(f"Generating embedding for text: {text[:100]}...")
                embedding = await self.ollama.create_embedding(text)
                
                if not embedding:
                    logger.warning(f"Empty embedding for text: {text[:50]}...")
                    embeddings.append([0.0] * embedding_dim)
                    continue
                    
                # Convert to list if it's a numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                    
                # Ensure correct dimension
                if len(embedding) != embedding_dim:
                    logger.warning(f"Embedding dimension mismatch: expected {embedding_dim}, got {len(embedding)}")
                    if len(embedding) > embedding_dim:
                        embedding = embedding[:embedding_dim]
                    else:
                        embedding.extend([0.0] * (embedding_dim - len(embedding)))
                
                # Debug the raw embedding values
                logger.debug(f"Raw embedding values (first 5): {embedding[:5]}")
                
                # Normalize
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding = (embedding_array / norm).tolist()
                    logger.debug(f"Normalized embedding (first 5): {embedding[:5]}")
                else:
                    logger.warning("Zero norm embedding detected")
                    
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}", exc_info=True)
                embeddings.append([0.0] * embedding_dim)
                
        return embeddings

class RAGService:
    def __init__(self):
        self.ollama = OllamaService()
        self.embedding_fn = OllamaEmbeddingFunction(self.ollama)
        self.collection_name = "documents"
        self.client = None
        self._embedding_dim = None

    async def initialize(self):
        """Initialize the Qdrant client and collection"""
        if self.client is None:
            self.client = await get_async_qdrant_client()
            self._embedding_dim = await self.embedding_fn._get_embedding_dimension()

        # Check if collection exists and recreate if needed
        try:
            collection_exists = await self.client.collection_exists(self.collection_name)
            
            if collection_exists:
                # Get collection info to check dimensions
                collection_info = await self.client.get_collection_info(self.collection_name)
                if collection_info and collection_info.vector_size != self._embedding_dim:
                    logger.warning(f"Collection dimension mismatch: expected {self._embedding_dim}, got {collection_info.vector_size}. Recreating...")
                    collection_exists = False  # Force recreation
                else:
                    logger.info(f"Collection {self.collection_name} already exists with correct dimensions")
                    return
        except Exception as e:
            logger.info(f"Collection check failed (may not exist): {str(e)}")
            collection_exists = False

        # Create collection with proper parameters
        try:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vector_size=self._embedding_dim,
                distance="Cosine",
                recreate_if_exists=True
            )
            logger.info(f"Created collection {self.collection_name} with dimension {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise

    async def _ensure_collection_exists(self):
        """Ensure collection exists with proper configuration"""
        try:
            # Check if client is initialized
            if self.client is None:
                await self.initialize()
                return
                
            # Verify collection exists
            collection_exists = await self.client.collection_exists(self.collection_name)
            if not collection_exists:
                await self.initialize()
                
        except Exception as e:
            logger.info(f"Collection verification failed, initializing: {str(e)}")
            await self.initialize()

    async def ingest_document(
        self,
        db: AsyncSession,
        user_id: str,
        filename: str,
        content: bytes,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> str:
        """Ingest document into Qdrant"""
        await self._ensure_collection_exists()
        
        try:
            logger.info(f"Starting document ingestion for {filename}")
            text = process_document(filename, content)
            if not text:
                raise ValueError("No text extracted")
            
            logger.info(f"Extracted text length: {len(text)}")
            
            chunks = chunk_text_with_overlap(
                text, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            if not chunks:
                raise ValueError("No chunks created")
            
            logger.info(f"Created {len(chunks)} chunks")
            
            doc_id = f"{user_id}_{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}"
            
            # Generate embeddings asynchronously
            logger.info("Generating embeddings...")
            embeddings = await self.embedding_fn.generate_embeddings(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Prepare points for upsert
            points = [
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "filename": filename,
                        "chunk_index": i,
                        "doc_id": doc_id,
                        "text": chunk,
                        "chunk_size": len(chunk),
                        "created_at": asyncio.get_event_loop().time()
                    }
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            # Upsert points
            success = await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            if not success:
                raise Exception("Failed to upsert points to Qdrant")
            
            logger.info(f"Successfully ingested document {filename} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Ingest failed: {str(e)}", exc_info=True)
            raise

    def _calculate_similarity_from_score(self, score: float) -> float:
        """Convert Qdrant cosine similarity score to similarity percentage"""
        # Qdrant cosine similarity returns values between -1 and 1
        # Convert to 0-1 range for easier interpretation
        return (score + 1) / 2
            
    async def query(
        self,
        db: AsyncSession,
        user_id: str,
        query: str,
        max_results: int = 5,
        min_score: float = 0.3  # Qdrant's cosine similarity threshold
    ) -> Dict[str, Any]:
        """Query documents using RAG with Qdrant"""
        await self._ensure_collection_exists()
        
        try:
            if not query or len(query.strip()) < 3:
                raise ValueError("Query must be at least 3 characters")

            logger.info(f"Processing query: {query}")
            
            # Generate query embedding
            logger.info("Generating query embedding...")
            query_embeddings = await self.embedding_fn.generate_embeddings([query])
            
            if not query_embeddings or len(query_embeddings[0]) == 0:
                raise ValueError("Failed to generate query embedding")
            
            logger.info(f"Query embedding dimension: {len(query_embeddings[0])}")

            # Import the filter creation function
            from app.utils.qdrant_async import create_user_filter
            
            # Search Qdrant
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embeddings[0],
                query_filter=create_user_filter(user_id),
                limit=max_results * 2,  # Get more results to filter
                score_threshold=min_score
            )
            
            if not search_results:
                return {
                    "answer": "No relevant documents found for your query",
                    "documents": [],
                    "context": [],
                    "sources": []
                }
            
            # Process results
            filtered_docs = []
            filtered_metas = []
            similarity_scores = []
            
            for result in search_results:
                similarity = self._calculate_similarity_from_score(result.score)
                similarity_scores.append(similarity)
                
                if similarity >= min_score:
                    filtered_docs.append(result.payload.get("text", ""))
                    filtered_metas.append(result.payload)
            
            logger.info(f"Filtered {len(filtered_docs)} documents from {len(search_results)}")
            
            if not filtered_docs:
                max_similarity = max(similarity_scores) if similarity_scores else 0.0
                return {
                    "answer": f"No relevant documents found above similarity threshold {min_score}. Maximum similarity found: {max_similarity:.4f}",
                    "documents": [],
                    "context": [],
                    "sources": []
                }
            
            # Build context
            context = "\n\n---\n\n".join([
                f"Source: {res.get('filename', 'Unknown')} (Chunk {res.get('chunk_index', 0) + 1})\n{doc}"
                for doc, res in zip(filtered_docs, filtered_metas)
            ])
            
            system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer based on the provided context, say so honestly. Don't make up information.
Be comprehensive but concise in your response.

Context:
{context}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Cite relevant sources when appropriate
- Be helpful and informative"""

            # Get answer from LLM
            logger.info("Generating LLM response...")
            llm_response = await self.ollama.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            return {
                "answer": llm_response.get("message", {}).get("content", "No response generated"),
                "documents": filtered_docs,
                "context": llm_response.get("context", []),
                "sources": list(set(
                    meta.get("filename", "Unknown") 
                    for meta in filtered_metas 
                    if meta and "filename" in meta
                )),
                "debug_info": {
                    "similarity_scores": similarity_scores[:5],
                    "threshold_used": min_score,
                    "total_chunks_found": len(search_results),
                    "chunks_after_filtering": len(filtered_docs),
                    "query_embedding_dim": len(query_embeddings[0])
                }
            }
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}", exc_info=True)
            raise

    async def list_documents(
        self,
        db: AsyncSession,
        user_id: str,
        page: int = 1,
        per_page: int = 10
    ) -> List[Dict[str, Any]]:
        """List documents with pagination"""
        await self._ensure_collection_exists()
        
        try:
            if page < 1 or per_page < 1:
                raise ValueError("Page and per_page must be positive integers")

            # Import the filter creation function
            from app.utils.qdrant_async import create_user_filter
            
            # Scroll through user's documents
            records, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=create_user_filter(user_id),
                with_payload=True,
                limit=1000  # Large number to get all docs
            )
            
            if not records:
                return []

            # Get unique documents with stats
            unique_docs = {}
            for record in records:
                payload = record.payload
                if payload and "doc_id" in payload:
                    doc_id = payload["doc_id"]
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = {
                            "document_id": doc_id,
                            "filename": payload.get("filename", "unknown"),
                            "chunk_count": 0,
                            "total_text_length": 0,
                            "created_at": payload.get("created_at", 0)
                        }
                    
                    unique_docs[doc_id]["chunk_count"] += 1
                    unique_docs[doc_id]["total_text_length"] += len(payload.get("text", ""))

            # Convert to list and sort by creation time
            doc_list = list(unique_docs.values())
            doc_list.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            
            # Apply pagination
            start = (page - 1) * per_page
            end = start + per_page
            return doc_list[start:end]
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}", exc_info=True)
            raise

    async def delete_document(
        self,
        db: AsyncSession,
        user_id: str,
        document_id: str
    ) -> bool:
        """Delete a document and all its chunks"""
        await self._ensure_collection_exists()
        
        try:
            # Import the filter creation function
            from app.utils.qdrant_async import create_document_filter
            
            # Delete points matching the document_id and user_id
            success = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=create_document_filter(user_id, document_id)
                )
            )
            
            logger.info(f"Successfully deleted document {document_id} for user {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}", exc_info=True)
            raise

    async def get_document_stats(self, db: AsyncSession, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's documents"""
        await self._ensure_collection_exists()
        
        try:
            # Import the filter creation function
            from app.utils.qdrant_async import create_user_filter
            
            # Count total points for user
            total_chunks = await self.client.count_points(
                collection_name=self.collection_name,
                count_filter=create_user_filter(user_id)
            )
            
            # Get unique documents
            records, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=create_user_filter(user_id),
                with_payload=True,
                limit=1000
            )
            
            unique_docs = set()
            total_text_length = 0
            
            for record in records:
                payload = record.payload
                if payload:
                    unique_docs.add(payload.get("doc_id", ""))
                    total_text_length += len(payload.get("text", ""))
            
            return {
                "total_documents": len(unique_docs),
                "total_chunks": total_chunks,
                "total_text_length": total_text_length,
                "average_chunk_size": total_text_length // max(total_chunks, 1),
                "embedding_dimension": self._embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}", exc_info=True)
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_text_length": 0,
                "average_chunk_size": 0,
                "embedding_dimension": self._embedding_dim or 384
            }

    async def reset_collection(self):
        """Reset the collection - useful for debugging"""
        await self.initialize()
        try:
            # Import the filter creation function
            from app.utils.qdrant_async import create_user_filter
            
            # Delete all points in the collection by creating an empty filter
            success = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(must=[])
                )
            )
            logger.info(f"Collection {self.collection_name} reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise

    async def close(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()
            logger.info("RAG service closed successfully")