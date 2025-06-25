# AI Agent Platform with Ollama

A multi-tenant AI agent platform using Ollama for local LLM interactions.

## Features

- Create and manage custom AI agents
- Chat with agents using different LLM models
- Document ingestion and RAG capabilities
- Voice interaction (STT/TTS)
- Agent training with custom data

## Setup

1. Install Ollama: https://ollama.ai/
2. Pull desired models: `ollama pull llama3`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the service: `uvicorn app.main:app --reload`

## API Documentation

Available at `/docs` when the service is running.





mkdir -p llmmodel/app/{routers,services,models,utils} llmmodel/tests && \
touch llmmodel/app/__init__.py \
      llmmodel/app/main.py \
      llmmodel/app/config.py \
      llmmodel/app/routers/__init__.py \
      llmmodel/app/routers/agents.py \
      llmmodel/app/routers/chat.py \
      llmmodel/app/routers/rag.py \
      llmmodel/app/routers/voice.py \
      llmmodel/app/routers/training.py \
      llmmodel/app/services/llm_service.py \
      llmmodel/app/services/rag_service.py \
      llmmodel/app/services/voice_service.py \
      llmmodel/app/services/agent_service.py \
      llmmodel/app/services/training_service.py \
      llmmodel/app/models/agent_model.py \
      llmmodel/app/models/response_schema.py \
      llmmodel/app/models/voice_model.py \
      llmmodel/app/models/training_model.py \
      llmmodel/app/utils/file_processing.py \
      llmmodel/app/utils/auth.py \
      llmmodel/app/utils/helpers.py \
      llmmodel/app/dependencies.py \
      llmmodel/tests/__init__.py \
      llmmodel/tests/test_chat.py \
      llmmodel/tests/test_rag.py \
      llmmodel/tests/test_voice.py \
      llmmodel/tests/test_agents.py \
      llmmodel/Dockerfile \
      llmmodel/requirements.txt \
      llmmodel/README.md# llm-model
