# app/services/rag_service.py
import logging
import os
import asyncio
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.utils.file_processing import process_document
from app.services.llm_service import OllamaService
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.qdrant_async import get_async_qdrant_client
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words
import re
from datetime import datetime
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

class OllamaEmbeddingFunction:
    def __init__(self, ollama_service: OllamaService):
        self.ollama = ollama_service
        self._embedding_dim = None
        self._model_checked = False
    
    async def _verify_embedding_model(self):
        """Verify the embedding model produces consistent dimensions"""
        if self._model_checked:
            return
            
        try:
            # Test with multiple inputs to verify consistency
            test_texts = ["test", "another test", "embedding verification"]
            embeddings = []
            
            for text in test_texts:
                embedding = await self.ollama.create_embedding(text)
                if not embedding:
                    raise ValueError(f"Empty embedding returned for text: {text}")
                embeddings.append(embedding)
            
            # Check all embeddings have same dimension
            dims = {len(e) for e in embeddings}
            if len(dims) != 1:
                raise ValueError(f"Inconsistent embedding dimensions: {dims}")
                
            self._embedding_dim = dims.pop()
            logger.info(f"Verified embedding dimension: {self._embedding_dim}")
            self._model_checked = True
            
        except Exception as e:
            logger.error(f"Embedding model verification failed: {str(e)}")
            raise

    async def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model"""
        if self._embedding_dim is None:
            await self._verify_embedding_model()
        return self._embedding_dim
    
    async def generate_embeddings(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings with dimension validation"""
        if not input:
            return []
            
        embedding_dim = await self._get_embedding_dimension()
        embeddings = []
        
        for text in input:
            try:
                logger.debug(f"Generating embedding for text: {text[:100]}...")
                embedding = await self.ollama.create_embedding(text)
                
                if not embedding:
                    logger.warning(f"Empty embedding for text: {text[:50]}...")
                    embeddings.append([0.0] * embedding_dim)
                    continue
                    
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                    
                # Validate dimension
                if len(embedding) != embedding_dim:
                    logger.warning(f"Embedding dimension mismatch: expected {embedding_dim}, got {len(embedding)}")
                    if len(embedding) > embedding_dim:
                        embedding = embedding[:embedding_dim]
                    else:
                        embedding.extend([0.0] * (embedding_dim - len(embedding)))
                
                # Normalize
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding = (embedding_array / norm).tolist()
                else:
                    logger.warning("Zero norm embedding detected")
                    
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}", exc_info=True)
                embeddings.append([0.0] * embedding_dim)
                
        return embeddings

class HybridRetriever:
    def __init__(self, ollama_service: OllamaService):
        self.ollama = ollama_service
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=list(_stop_words.ENGLISH_STOP_WORDS),
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.cross_encoder = None
        self._is_tfidf_trained = False
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize cross-encoder if enabled"""
        if settings.RERANKING_ENABLED:
            try:
                self.cross_encoder = CrossEncoder(settings.RERANKING_MODEL)
                logger.info(f"Initialized cross-encoder: {settings.RERANKING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize cross-encoder: {str(e)}")
                self.cross_encoder = None
    
    async def train_tfidf(self, corpus: List[str]):
        """Train TF-IDF vectorizer on a corpus of documents"""
        if not corpus:
            logger.warning("No documents provided for TF-IDF training")
            return
            
        try:
            logger.info(f"Training TF-IDF on {len(corpus)} documents...")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            self._is_tfidf_trained = True
            logger.info(f"TF-IDF vectorizer trained with {len(self.tfidf_vectorizer.get_feature_names_out())} features")
        except Exception as e:
            logger.error(f"Failed to train TF-IDF vectorizer: {str(e)}")
            self._is_tfidf_trained = False
    
    def get_sparse_embedding(self, text: str) -> Optional[List[float]]:
        """Generate sparse TF-IDF embedding with dimension validation"""
        if not self._is_tfidf_trained:
            return None
            
        try:
            vector = self.tfidf_vectorizer.transform([text])
            return vector.toarray()[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate sparse embedding: {str(e)}")
            return None

    async def rerank_results(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """Re-rank results using cross-encoder"""
        if not self.cross_encoder or not documents:
            return documents, scores
            
        if len(documents) != len(scores):
            logger.error(f"Mismatched documents ({len(documents)}) and scores ({len(scores)})")
            return documents, scores
            
        try:
            # Create query-document pairs for cross-encoder
            pairs = [(query, doc) for doc in documents]
            
            # Get scores from cross-encoder
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Combine with original scores
            combined_scores = [
                (settings.HYBRID_DENSE_WEIGHT * score) + 
                (settings.HYBRID_SPARSE_WEIGHT * rerank_score)
                for score, rerank_score in zip(scores, rerank_scores)
            ]
            
            # Sort documents by combined scores
            sorted_indices = np.argsort(combined_scores)[::-1]
            sorted_docs = [documents[i] for i in sorted_indices[:top_k]]
            sorted_scores = [combined_scores[i] for i in sorted_indices[:top_k]]
            
            return sorted_docs, sorted_scores
        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            return documents, scores

class QueryRewriter:
    @staticmethod
    async def expand_query(query: str, ollama_service: OllamaService) -> str:
        """Expand query using LLM to generate synonyms and related terms"""
        if not settings.QUERY_REWRITING_ENABLED:
            return query
            
        try:
            prompt = f"""Expand this search query with synonyms and related terms. 
Return only a comma-separated list of terms without any additional text.

Original query: {query}

Expanded terms:"""
            
            response = await ollama_service.chat(
                messages=[{"role": "user", "content": prompt}],
                model=settings.DEFAULT_OLLAMA_MODEL,
                temperature=0.7,
                max_tokens=100
            )
            
            expanded_terms = response.get("message", {}).get("content", "").strip()
            if expanded_terms:
                expanded_terms = re.sub(r'[^a-zA-Z0-9,\s]', '', expanded_terms)
                expanded_terms = ', '.join([term.strip() for term in expanded_terms.split(',') if term.strip()])
                return f"{query}, {expanded_terms}"
            return query
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return query
    
    @staticmethod
    async def rewrite_query(query: str, ollama_service: OllamaService) -> str:
        """Rewrite query for better retrieval using LLM"""
        if not settings.QUERY_REWRITING_ENABLED:
            return query
            
        try:
            prompt = f"""Rewrite this search query to be more effective for document retrieval. 
Keep the original meaning but optimize for semantic search. 
Return only the rewritten query without any additional text.

Original query: {query}

Rewritten query:"""
            
            response = await ollama_service.chat(
                messages=[{"role": "user", "content": prompt}],
                model=settings.DEFAULT_OLLAMA_MODEL,
                temperature=0.3,
                max_tokens=100
            )
            
            rewritten = response.get("message", {}).get("content", "").strip()
            return rewritten if rewritten else query
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            return query

class RAGService:
    def __init__(self):
        self.ollama = OllamaService()
        self.embedding_fn = OllamaEmbeddingFunction(self.ollama)
        self.hybrid_retriever = HybridRetriever(self.ollama)
        self.query_rewriter = QueryRewriter()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.client = None
        self._embedding_dim = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the Qdrant client and collection with proper dimension handling"""
        async with self._lock:
            if self._initialized:
                return

            try:
                self.client = await get_async_qdrant_client()
                
                # Verify embedding model first
                await self.embedding_fn._verify_embedding_model()
                self._embedding_dim = await self.embedding_fn._get_embedding_dimension()

                # Check if collection exists and has correct dimensions
                collection_exists = await self.client.collection_exists(self.collection_name)
                needs_recreation = False
                
                if collection_exists:
                    collection_info = await self.client.get_collection_info(self.collection_name)
                    if collection_info:
                        if collection_info.vector_size != self._embedding_dim:
                            logger.warning(
                                f"Collection dimension mismatch: "
                                f"expected {self._embedding_dim}, got {collection_info.vector_size}. "
                                "Recreating collection..."
                            )
                            needs_recreation = True
                    else:
                        logger.warning("Could not get collection info, assuming recreation needed")
                        needs_recreation = True

                if not collection_exists or needs_recreation:
                    await self.client.create_collection(
                        collection_name=self.collection_name,
                        vector_size=self._embedding_dim,
                        distance="Cosine",
                        recreate_if_exists=True
                    )
                    logger.info(f"Created collection {self.collection_name} with dimension {self._embedding_dim}")

                # Initialize TF-IDF with sample data
                sample_docs = [
                    "Artificial intelligence is transforming industries.",
                    "Machine learning models require large datasets.",
                    "Natural language processing enables text understanding.",
                    "Deep learning uses neural networks for pattern recognition."
                ]
                await self.hybrid_retriever.train_tfidf(sample_docs)

                self._initialized = True
                logger.info("RAGService initialized successfully")

            except Exception as e:
                logger.error(f"Initialization failed: {str(e)}", exc_info=True)
                raise

    async def _ensure_collection_exists(self):
        """Ensure collection exists with proper configuration"""
        if not self._initialized:
            await self.initialize()



    def _calculate_similarity_from_score(self, score: float) -> float:
        """Convert Qdrant cosine similarity score to similarity percentage"""
        return max(0.0, min(1.0, (score + 1) / 2)) 

    async def ingest_document(
        self,
        db: AsyncSession,
        user_id: str,
        agent_id: str,
        filename: str,
        content: bytes,
        chunk_size: int = None,
        chunk_overlap: int = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Ingest document into Qdrant with proper dimension handling"""
        await self._ensure_collection_exists()
        
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
        
        try:
            logger.info(f"Starting document ingestion for {filename}")
            text = process_document(filename, content)
            if not text:
                raise ValueError("No text extracted from document")
            
            # Generate document ID with hash for uniqueness
            doc_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            doc_id = f"{user_id}_{agent_id}_{os.path.splitext(filename)[0]}_{doc_hash}"
            
            # Process metadata with agent_id
            processed_metadata = {
                "user_id": user_id,
                "agent_id": agent_id,
                "filename": filename,
                "doc_id": doc_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            if metadata:
                processed_metadata.update(metadata)
            
            # Chunk the document
            chunks = self._chunk_text_with_overlap(
                text, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            if not chunks:
                raise ValueError("No valid chunks created from document")
            
            # Generate embeddings
            embeddings = await self.embedding_fn.generate_embeddings(chunks)
            
            # Verify embedding dimensions match collection
            collection_info = await self.client.get_collection_info(self.collection_name)
            if not collection_info:
                raise ValueError("Could not get collection information")
                
            for i, embedding in enumerate(embeddings):
                if len(embedding) != collection_info.vector_size:
                    raise ValueError(
                        f"Embedding dimension mismatch at chunk {i}: "
                        f"expected {collection_info.vector_size}, got {len(embedding)}"
                    )
            
            # Prepare points for upsert
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = processed_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "text_length": len(chunk),
                    "created_at": datetime.utcnow().isoformat()
                })
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{i}"))
                
                points.append(qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        **chunk_metadata
                    }
                ))
            
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

    def _chunk_text_with_overlap(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int,
        separator: str = "\n\n"
    ) -> List[str]:
        """Improved text chunking with semantic boundaries"""
        if not text or not text.strip():
            return []
        
        if len(text) <= chunk_size:
            return [text.strip()]
        
        # First split by major sections
        sections = re.split(r'\n{2,}', text)
        chunks = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if len(section) <= chunk_size:
                chunks.append(section)
                continue
                
            # Then split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) + 1 > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        
                        # Keep overlap if specified
                        if chunk_overlap > 0:
                            overlap_start = max(0, len(current_chunk) - chunk_overlap)
                            overlap_text = current_chunk[overlap_start:]
                            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Sentence is too long, split by words
                        words = sentence.split()
                        current_word_chunk = []
                        
                        for word in words:
                            if len(" ".join(current_word_chunk + [word])) > chunk_size:
                                if current_word_chunk:
                                    chunks.append(" ".join(current_word_chunk))
                                    current_word_chunk = current_word_chunk[-chunk_overlap:] if chunk_overlap > 0 else []
                                current_word_chunk.append(word)
                            else:
                                current_word_chunk.append(word)
                                
                        if current_word_chunk:
                            current_chunk = " ".join(current_word_chunk)
                else:
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if current_chunk:
                chunks.append(current_chunk)
        
        # Post-processing to clean up chunks
        cleaned_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or len(chunk) < 50:  # Minimum chunk size
                continue
                
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            if chunk_hash not in seen_hashes:
                cleaned_chunks.append(chunk)
                seen_hashes.add(chunk_hash)
        
        logger.info(f"Split text into {len(cleaned_chunks)} chunks with overlap")
        return cleaned_chunks

    async def query(
        self,
        db: AsyncSession,
        user_id: str,
        agent_id: str,
        query: str,
        max_results: int = 5,
        min_score: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
        rewrite_query: bool = True,
        use_reranking: bool = True,
        hybrid_search: bool = True
    ) -> Dict[str, Any]:
        """Enhanced query with hybrid search and query rewriting"""
        await self._ensure_collection_exists()
        
        try:
            # Validate input
            if not query or len(query.strip()) < 3:
                raise ValueError("Query must be at least 3 characters")
            
            max_results = min(max(1, max_results), 20)  # Clamp between 1-20
            min_score = max(0.0, min(1.0, min_score))  # Clamp between 0-1

            # Query processing
            processed_query = await self._process_query(query, rewrite_query)
            
            # Search execution with agent_id filter
            search_results = await self._execute_search(
                user_id, 
                agent_id,
                processed_query, 
                max_results * 3,  # Get more results for re-ranking
                min_score,
                filters,
                hybrid_search
            )
            
            if not search_results:
                return self._empty_response(query)
            
            # Re-ranking if enabled
            if use_reranking and settings.RERANKING_ENABLED:
                search_results = await self._rerank_results(processed_query, search_results, max_results)
            else:
                search_results = search_results[:max_results]
            
            # Response generation
            return await self._generate_response(query, processed_query, search_results, use_reranking, hybrid_search)
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}", exc_info=True)
            raise

    async def _process_query(self, query: str, rewrite: bool) -> str:
        """Process and optimize the query"""
        processed_query = query
        
        if rewrite and settings.QUERY_REWRITING_ENABLED:
            processed_query = await self.query_rewriter.rewrite_query(query, self.ollama)
            logger.info(f"Rewritten query: {processed_query}")
            
            if settings.QUERY_EXPANSION_ENABLED:
                expanded_query = await self.query_rewriter.expand_query(processed_query, self.ollama)
                if expanded_query != processed_query:
                    logger.info(f"Expanded query: {expanded_query}")
                    processed_query = expanded_query
        
        return processed_query

    async def _execute_search(
        self,
        user_id: str,
        agent_id: str,
        query: str,
        limit: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
        hybrid_search: bool
    ) -> List[qdrant_models.ScoredPoint]:
        """Execute the search with hybrid approach if enabled"""
        from app.utils.qdrant_async import create_user_filter
        
        # Get collection info for dimension verification
        collection_info = await self.client.get_collection_info(self.collection_name)
        if not collection_info:
            raise ValueError("Could not get collection information")

        # Generate dense embeddings
        query_embeddings = await self.embedding_fn.generate_embeddings([query])
        if not query_embeddings or not query_embeddings[0]:
            raise ValueError("Failed to generate query embedding")
        
        # Verify embedding dimension matches collection
        if len(query_embeddings[0]) != collection_info.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch: query has {len(query_embeddings[0])}, "
                f"collection expects {collection_info.vector_size}"
            )

        # Add agent_id to filters
        if filters is None:
            filters = {}
        filters["agent_id"] = agent_id

        # Dense vector search
        dense_results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embeddings[0],
            query_filter=create_user_filter(user_id, filters),
            limit=limit,
            score_threshold=min_score
        )
        
        # Early return if hybrid search is disabled
        if not hybrid_search or not settings.HYBRID_SEARCH_ENABLED:
            return dense_results
        
        # Sparse search (TF-IDF)
        sparse_embedding = self.hybrid_retriever.get_sparse_embedding(query)
        sparse_results = []
        if sparse_embedding:
            try:
                # Qdrant requires all vectors to have same dimension as collection
                # So we need to pad/truncate sparse embeddings to match
                sparse_embedding_adjusted = sparse_embedding.copy()
                if len(sparse_embedding_adjusted) > collection_info.vector_size:
                    sparse_embedding_adjusted = sparse_embedding_adjusted[:collection_info.vector_size]
                else:
                    sparse_embedding_adjusted.extend(
                        [0.0] * (collection_info.vector_size - len(sparse_embedding_adjusted))
                    )
                
                sparse_results = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=sparse_embedding_adjusted,
                    query_filter=create_user_filter(user_id, filters),
                    limit=limit,
                    score_threshold=min_score
                )
            except Exception as e:
                logger.error(f"Sparse search failed: {str(e)}")
                # Fall back to dense results only
                return dense_results
        
        return self._combine_results(dense_results, sparse_results)

    def _combine_results(
        self,
        dense_results: List[qdrant_models.ScoredPoint],
        sparse_results: List[qdrant_models.ScoredPoint]
    ) -> List[qdrant_models.ScoredPoint]:
        """Combine dense and sparse search results"""
        if not sparse_results or not settings.HYBRID_SEARCH_ENABLED:
            return dense_results
            
        # Create a map of unique documents
        results_map = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result.payload.get("doc_id", str(result.id))
            results_map[doc_id] = {
                "result": result,
                "dense_score": result.score,
                "sparse_score": 0.0
            }
        
        # Process sparse results
        for result in sparse_results:
            doc_id = result.payload.get("doc_id", str(result.id))
            if doc_id in results_map:
                results_map[doc_id]["sparse_score"] = result.score
            else:
                results_map[doc_id] = {
                    "result": result,
                    "dense_score": 0.0,
                    "sparse_score": result.score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_data in results_map.values():
            combined_score = (
                settings.HYBRID_DENSE_WEIGHT * doc_data["dense_score"] +
                settings.HYBRID_SPARSE_WEIGHT * doc_data["sparse_score"]
            )
            new_result = doc_data["result"]
            new_result.score = combined_score
            combined_results.append(new_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results

    async def _rerank_results(
        self,
        query: str,
        results: List[qdrant_models.ScoredPoint],
        top_k: int
    ) -> List[qdrant_models.ScoredPoint]:
        """Re-rank results using cross-encoder"""
        if not results or not settings.RERANKING_ENABLED:
            return results[:top_k]
            
        documents = [res.payload.get("text", "") for res in results]
        scores = [res.score for res in results]
        
        reranked_docs, reranked_scores = await self.hybrid_retriever.rerank_results(
            query,
            documents,
            scores,
            top_k
        )
        
        # Reconstruct results with new order
        reranked_results = []
        doc_set = set(reranked_docs)
        for doc, score in zip(reranked_docs, reranked_scores):
            for res in results:
                if res.payload.get("text", "") == doc and res not in reranked_results:
                    res.score = score  # Update score
                    reranked_results.append(res)
                    break
        
        return reranked_results[:top_k]

    async def _generate_response(
        self,
        original_query: str,
        processed_query: str,
        results: List[qdrant_models.ScoredPoint],
        use_reranking: bool,
        hybrid_search_used: bool
    ) -> Dict[str, Any]:
        """Generate final response with LLM"""
        if not results:
            return self._empty_response(original_query)
        
        # Extract documents and metadata
        documents = []
        metadatas = []
        for result in results:
            documents.append(result.payload.get("text", ""))
            metadatas.append(result.payload)
        
        # Build context
        context = "\n\n---\n\n".join([
            f"Source: {meta.get('filename', 'Unknown')} (Chunk {meta.get('chunk_index', 0) + 1})\n{doc}"
            for doc, meta in zip(documents, metadatas)
        ])
        
        # Generate answer with LLM
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

        llm_response = await self.ollama.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query}  # Use original query here
            ]
        )
        
        # Prepare response
        sources = list(set(
            meta.get("filename", "Unknown") 
            for meta in metadatas 
            if meta and "filename" in meta
        ))
        
        return {
            "answer": llm_response.get("message", {}).get("content", "No response generated"),
            "documents": documents,
            "context": llm_response.get("context", []),
            "sources": sources,
            "debug_info": {
                "original_query": original_query,
                "processed_query": processed_query,
                "search_method": "hybrid" if hybrid_search_used else "dense",
                "reranking_applied": use_reranking,
                "total_results": len(results),
                "scores": [self._calculate_similarity_from_score(r.score) for r in results],
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _empty_response(self, query: str) -> Dict[str, Any]:
        """Generate empty response structure"""
        return {
            "answer": f"No relevant documents found for your query: {query}",
            "documents": [],
            "context": [],
            "sources": [],
            "debug_info": {
                "original_query": query,
                "search_method": "none",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def list_documents(
        self,
        db: AsyncSession,
        user_id: str,
        agent_id: str,
        page: int = 1,
        per_page: int = 10
    ) -> List[Dict[str, Any]]:
        """List documents with pagination"""
        await self._ensure_collection_exists()
        
        try:
            if page < 1 or per_page < 1:
                raise ValueError("Page and per_page must be positive integers")

            from app.utils.qdrant_async import create_user_filter
            
            # Include agent_id in filter
            filters = {"agent_id": agent_id}
            
            records, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=create_user_filter(user_id, filters),
                with_payload=True,
                limit=1000
            )
            
            if not records:
                return []

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

            doc_list = list(unique_docs.values())
            doc_list.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            
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
            from app.utils.qdrant_async import create_document_filter
            
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
            from app.utils.qdrant_async import create_user_filter
            
            total_chunks = await self.client.count_points(
                collection_name=self.collection_name,
                count_filter=create_user_filter(user_id)
            )
            
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

    async def close(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()
        logger.info("RAG service closed successfully")