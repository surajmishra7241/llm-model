# ./app/services/enhanced_agent_service.py
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
import time
from app.services.llm_service import OllamaService
from app.services.rag_service import RAGService

class ReasoningMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced" 
    DEEP = "deep"

class ResponseFormat(str, Enum):
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    JSON = "json"

class EnhancedAgentExecutor:
    def __init__(self):
        self.llm_service = OllamaService()
        self.rag_service = RAGService()
    
    async def execute_with_reasoning(
        self,
        agent,
        input_text: str,
        parameters: Dict[str, Any],
        reasoning_mode: ReasoningMode = ReasoningMode.BALANCED
    ) -> Dict[str, Any]:
        """Execute agent with enhanced reasoning capabilities"""
        
        reasoning_steps = []
        start_time = time.time()
        
        try:
            # Step 1: Analyze intent and complexity
            if reasoning_mode == ReasoningMode.DEEP:
                intent_analysis = await self._analyze_intent(input_text)
                reasoning_steps.append(f"Intent Analysis: {intent_analysis}")
            
            # Step 2: Gather context (RAG, memory, etc.)
            context = await self._gather_context(agent, input_text, parameters)
            reasoning_steps.append(f"Context gathered: {len(context.get('documents', []))} relevant documents")
            
            # Step 3: Generate response with reasoning
            if reasoning_mode == ReasoningMode.DEEP:
                # Multi-step reasoning for complex queries
                response = await self._deep_reasoning_response(agent, input_text, context, parameters)
            elif reasoning_mode == ReasoningMode.FAST:
                # Single-step for quick responses
                response = await self._fast_response(agent, input_text, context, parameters)
            else:
                # Balanced approach
                response = await self._balanced_response(agent, input_text, context, parameters)
            
            execution_time = time.time() - start_time
            
            return {
                "response": response.get("content", ""),
                "reasoning_steps": reasoning_steps,
                "execution_time": execution_time,
                "sources": context.get("sources", []),
                "context_used": context.get("documents", [])[:3],  # Limit for response size
                "model": agent.model,
                "reasoning_mode": reasoning_mode.value
            }
            
        except Exception as e:
            logger.error(f"Enhanced execution failed: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "reasoning_steps": reasoning_steps,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def _analyze_intent(self, input_text: str) -> str:
        """Analyze user intent for complex reasoning"""
        prompt = f"""Analyze the intent and complexity of this user request:
        
        User Input: {input_text}
        
        Provide a brief analysis of:
        1. Primary intent (question, task, creative request, etc.)
        2. Complexity level (simple, moderate, complex)
        3. Required capabilities (reasoning, search, creativity, etc.)
        
        Keep response concise (2-3 sentences)."""
        
        response = await self.llm_service.generate(
            prompt=prompt,
            options={"temperature": 0.3, "max_tokens": 150}
        )
        return response.get("response", "Unable to analyze intent")
    
    async def _gather_context(self, agent, input_text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant context from multiple sources"""
        context = {"documents": [], "sources": []}
        
        # RAG context if enabled
        if parameters.get("enable_rag", True):
            try:
                await self.rag_service.initialize()
                rag_results = await self.rag_service.query(
                    db=None,  # This needs to be passed properly
                    user_id=parameters.get("user_id"),
                    agent_id=agent.id,
                    query=input_text,
                    max_results=5
                )
                context["documents"] = rag_results.get("documents", [])
                context["sources"] = rag_results.get("sources", [])
            except Exception as e:
                logger.warning(f"RAG context gathering failed: {str(e)}")
        
        return context
    
    async def _deep_reasoning_response(self, agent, input_text: str, context: Dict, parameters: Dict) -> Dict[str, Any]:
        """Generate response with deep reasoning (chain of thought)"""
        
        # Build enhanced system prompt for deep reasoning
        system_prompt = f"""{agent.system_prompt}

You are now in DEEP REASONING mode. For complex queries:
1. Break down the problem into steps
2. Show your thinking process
3. Use the provided context when relevant
4. Provide a comprehensive, well-reasoned response

Context Information:
{chr(10).join(context.get('documents', [])[:3])}

Think step by step and show your reasoning."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please think through this carefully: {input_text}"}
        ]
        
        response = await self.llm_service.chat(
            messages=messages,
            model=agent.model,
            options={
                "temperature": parameters.get("temperature", 0.7),
                "max_tokens": parameters.get("max_tokens", 3000),
            }
        )
        
        return response.get("message", {})
    
    async def _fast_response(self, agent, input_text: str, context: Dict, parameters: Dict) -> Dict[str, Any]:
        """Generate quick response for simple queries"""
        
        system_prompt = f"""{agent.system_prompt}

Provide a direct, concise response. Be helpful but brief."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        response = await self.llm_service.chat(
            messages=messages,
            model=agent.model,
            options={
                "temperature": parameters.get("temperature", 0.5),
                "max_tokens": min(parameters.get("max_tokens", 1000), 1000),
            }
        )
        
        return response.get("message", {})
    
    async def _balanced_response(self, agent, input_text: str, context: Dict, parameters: Dict) -> Dict[str, Any]:
        """Generate balanced response (default mode)"""
        
        context_text = ""
        if context.get("documents"):
            context_text = f"\n\nRelevant Context:\n{chr(10).join(context['documents'][:2])}"
        
        system_prompt = f"""{agent.system_prompt}

Use the following context if relevant to provide a helpful response.{context_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        response = await self.llm_service.chat(
            messages=messages,
            model=agent.model,
            options={
                "temperature": parameters.get("temperature", 0.7),
                "max_tokens": parameters.get("max_tokens", 2000),
            }
        )
        
        return response.get("message", {})

# Global instance
enhanced_executor = EnhancedAgentExecutor()
