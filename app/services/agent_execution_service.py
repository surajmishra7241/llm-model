from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.db_models import DBAgent
from app.services.llm_service import OllamaService
from app.services.cache import cache_service
from app.services.multi_search_service import MultiSearchService, SearchResultAggregator
from app.models.agent_model import AgentPersonality
import json
import logging
import hashlib
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

async def execute_agent(agent_id: str, user_id: str, input_data: dict, db: AsyncSession):
    """Enhanced agent execution with internet search capabilities"""
    
    start_time = time.time()
    
    try:
        # Get agent from database
        result = await db.execute(
            select(DBAgent).where(
                DBAgent.id == agent_id,
                DBAgent.owner_id == user_id
            )
        )
        agent = result.scalars().first()
        
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Extract parameters
        parameters = input_data.get("parameters", {})
        user_input = input_data.get("input", "")
        
        # Determine if internet search is needed
        enable_search = parameters.get("enable_search", True)
        search_sources = parameters.get("search_sources", ["duckduckgo", "reddit"])
        
        search_context = ""
        search_results_info = {}
        
        if enable_search and await _should_search_internet(user_input):
            try:
                # Perform multi-source search
                search_service = MultiSearchService()
                aggregator = SearchResultAggregator()
                
                logger.info(f"Performing internet search for query: {user_input[:50]}...")
                
                search_results = await search_service.search_all_sources(
                    query=user_input,
                    sources=search_sources,
                    max_results_per_source=parameters.get("max_search_results", 3)
                )
                
                # Aggregate and rank results
                aggregated_results = aggregator.aggregate_results(
                    search_results, 
                    max_final_results=8
                )
                
                # Build search context
                search_context = _build_search_context(aggregated_results)
                
                search_results_info = {
                    "performed": True,
                    "sources_used": list(search_results.keys()),
                    "total_results": len(aggregated_results),
                    "top_sources": [r.source for r in aggregated_results[:3]]
                }
                
                logger.info(f"Search completed: {len(aggregated_results)} results from {len(search_results)} sources")
                
            except Exception as search_error:
                logger.error(f"Internet search failed: {str(search_error)}")
                search_context = ""
                search_results_info = {
                    "performed": False,
                    "error": str(search_error)
                }
        else:
            search_results_info = {"performed": False, "reason": "Search not needed or disabled"}
        
        # Get conversation history
        history_key = f"conversation:{user_id}:{agent_id}"
        conversation_history = []
        
        try:
            cached_history = await cache_service.get(history_key)
            if cached_history:
                conversation_history = json.loads(cached_history)
        except Exception as cache_error:
            logger.warning(f"Failed to retrieve conversation history: {str(cache_error)}")
        
        # Build enhanced system prompt
        enhanced_prompt = _build_enhanced_system_prompt(agent, search_context)
        
        # Build conversation with personality context
        personality = AgentPersonality(**agent.personality) if agent.personality else AgentPersonality()
        
        messages = [
            {"role": "system", "content": enhanced_prompt},
            *conversation_history[-10:],  # Keep last 10 messages
            {"role": "user", "content": user_input}
        ]
        
        # Execute with LLM
        llm_service = OllamaService()
        
        llm_options = {
            "temperature": parameters.get("temperature", personality.base_tone_temperature),
            "top_p": parameters.get("top_p", 0.9),
            "max_tokens": parameters.get("max_tokens", 2000),
        }
        
        response = await llm_service.chat(
            messages=messages,
            model=agent.model,
            options=llm_options
        )
        
        # Extract response content
        response_content = ""
        if response and "message" in response and "content" in response["message"]:
            response_content = response["message"]["content"]
        else:
            response_content = "I apologize, but I encountered an issue generating a response."
        
        # Update conversation history
        new_user_message = {"role": "user", "content": user_input}
        new_assistant_message = {"role": "assistant", "content": response_content}
        
        updated_history = conversation_history + [new_user_message, new_assistant_message]
        
        # Keep only recent messages (last 20)
        max_history = personality.memory.context_window if personality.memory else 20
        updated_history = updated_history[-max_history:]
        
        # Cache updated conversation history
        try:
            history_ttl = personality.memory.memory_refresh_interval if personality.memory else 3600
            await cache_service.set(
                history_key,
                json.dumps(updated_history, default=str, ensure_ascii=False),
                ttl=history_ttl
            )
        except Exception as cache_error:
            logger.warning(f"Failed to cache conversation history: {str(cache_error)}")
        
        execution_time = time.time() - start_time
        
        return {
            "response": response_content,
            "model": agent.model,
            "personality_applied": personality.dict() if personality else {},
            "context_length": len(updated_history),
            "execution_time_ms": round(execution_time * 1000, 2),
            "search_info": search_results_info,
            "success": True
        }
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise ve
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
        execution_time = time.time() - start_time
        
        return {
            "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
            "model": "error",
            "personality_applied": {},
            "context_length": 0,
            "execution_time_ms": round(execution_time * 1000, 2),
            "search_info": {"performed": False, "error": str(e)},
            "success": False,
            "error": str(e)
        }

async def _should_search_internet(query: str) -> bool:
    """Determine if a query should trigger internet search"""
    
    # Skip very short queries
    if len(query.strip()) < 3:
        return False
    
    # Keywords that suggest current information is needed
    current_info_keywords = [
        'news', 'recent', 'latest', 'current', 'today', 'yesterday', 
        'this week', 'this month', 'this year', 'now', 'currently',
        'trending', 'popular', 'new', 'update', 'breaking'
    ]
    
    # Question words that often need current info
    question_keywords = [
        'what is', 'what are', 'who is', 'where is', 'when is',
        'how to', 'why is', 'which', 'compare', 'best', 'top'
    ]
    
    # Technical topics that benefit from search
    tech_keywords = [
        'error', 'bug', 'issue', 'problem', 'solution', 'fix',
        'tutorial', 'guide', 'documentation', 'api', 'code'
    ]
    
    query_lower = query.lower()
    
    # Check for indicators that search would be helpful
    for keyword_list in [current_info_keywords, question_keywords, tech_keywords]:
        if any(keyword in query_lower for keyword in keyword_list):
            return True
    
    # Check for specific patterns
    if any(pattern in query_lower for pattern in ['?', 'how do', 'what does', 'explain']):
        return True
    
    # Default to search for queries longer than 10 words (likely complex questions)
    if len(query.split()) > 10:
        return True
    
    return False

def _build_enhanced_system_prompt(agent, search_context: str) -> str:
    """Build enhanced system prompt with search context"""
    
    base_prompt = agent.system_prompt or "You are a helpful AI assistant."
    
    if search_context:
        enhanced_prompt = f"""{base_prompt}

You have access to recent information from the internet to help answer the user's question. Use this information to provide accurate, up-to-date, and comprehensive responses.

RECENT INTERNET INFORMATION:
{search_context}

Instructions:
1. Use the recent internet information when relevant to the user's question
2. Combine your existing knowledge with this new information
3. If the internet information contradicts your knowledge, prefer the more recent information
4. Always be helpful and provide specific, actionable information when possible
5. If you cite specific information from the search results, mention the source naturally
6. Don't mention that you performed a search - just provide the information naturally"""
    else:
        enhanced_prompt = base_prompt
    
    return enhanced_prompt

def _build_search_context(results: List) -> str:
    """Build search context from aggregated results"""
    if not results:
        return ""
    
    context_parts = []
    
    for i, result in enumerate(results[:6]):  # Use top 6 results
        source_info = f"[Source: {result.source.title()}]"
        if result.metadata and result.metadata.get('subreddit'):
            source_info += f" (r/{result.metadata['subreddit']})"
        
        context_parts.append(
            f"{i+1}. **{result.title}** {source_info}\n"
            f"   {result.content[:400]}{'...' if len(result.content) > 400 else ''}\n"
        )
    
    return "\n".join(context_parts)
