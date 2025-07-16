# LLM Model API Documentation

This document provides instructions on how to utilize the various APIs available in this project.

**Base URL:** `http://localhost:8000`

---

## Authentication

### Get OAuth2 Token

- **Endpoint:** `/token`
- **Method:** `POST`
- **Description:** Authenticates the user and returns a JWT token.
- **Payload:** `application/x-www-form-urlencoded`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/token" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "username=your_username&password=your_password"
```

---

## Agents

### Create Agent

- **Endpoint:** `/agents/`
- **Method:** `POST`
- **Description:** Creates a new agent.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/agents/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "name": "My Agent",
  "description": "A description of my agent.",
  "system_prompt": "You are a helpful assistant.",
  "tools": ["tool1", "tool2"],
  "llm": "default"
}'
```

### Get All Agents

- **Endpoint:** `/agents/`
- **Method:** `GET`
- **Description:** Retrieves all agents.

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/agents/" \
-H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Get Agent by ID

- **Endpoint:** `/agents/{agent_id}`
- **Method:** `GET`
- **Description:** Retrieves a specific agent by its ID.

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/agents/1" \
-H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Update Agent

- **Endpoint:** `/agents/{agent_id}`
- **Method:** `PUT`
- **Description:** Updates an existing agent.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X PUT "http://localhost:8000/agents/1" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "name": "Updated Agent Name",
  "description": "Updated description.",
  "system_prompt": "You are an updated helpful assistant.",
  "tools": ["tool1", "tool2", "tool3"],
  "llm": "updated_llm"
}'
```

### Delete Agent

- **Endpoint:** `/agents/{agent_id}`
- **Method:** `DELETE`
- **Description:** Deletes an agent by its ID.

**Curl Example:**
```bash
curl -X DELETE "http://localhost:8000/agents/1" \
-H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Agent Interaction

### Create Agent Interaction

- **Endpoint:** `/agent-interactions/`
- **Method:** `POST`
- **Description:** Creates a new agent interaction.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/agent-interactions/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "agent_id": 1,
  "session_id": "session123",
  "query": "What is the weather like today?",
  "response": "The weather is sunny."
}'
```

### Get All Agent Interactions

- **Endpoint:** `/agent-interactions/`
- **Method:** `GET`
- **Description:** Retrieves all agent interactions.

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/agent-interactions/" \
-H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Get Agent Interaction by ID

- **Endpoint:** `/agent-interactions/{interaction_id}`
- **Method:** `GET`
- **Description:** Retrieves a specific agent interaction by its ID.

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/agent-interactions/1" \
-H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Update Agent Interaction

- **Endpoint:** `/agent-interactions/{interaction_id}`
- **Method:** `PUT`
- **Description:** Updates an existing agent interaction.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X PUT "http://localhost:8000/agent-interactions/1" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "agent_id": 1,
  "session_id": "session123",
  "query": "What is the weather like tomorrow?",
  "response": "The weather will be rainy."
}'
```

### Delete Agent Interaction

- **Endpoint:** `/agent-interactions/{interaction_id}`
- **Method:** `DELETE`
- **Description:** Deletes an agent interaction by its ID.

**Curl Example:**
```bash
curl -X DELETE "http://localhost:8000/agent-interactions/1" \
-H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Chat

### Chat with Agent

- **Endpoint:** `/chat/`
- **Method:** `POST`
- **Description:** Sends a message to a chat agent and gets a response.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/chat/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "agent_id": 1,
  "query": "Hello, who are you?",
  "session_id": "session123"
}'
```

### Chat with Agent (Streaming)

- **Endpoint:** `/chat/stream`
- **Method:** `POST`
- **Description:** Sends a message to a chat agent and gets a streaming response.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/chat/stream" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "agent_id": 1,
  "query": "Hello, who are you?",
  "session_id": "session123"
}'
```

---

## Execute

### Execute Command

- **Endpoint:** `/execute/`
- **Method:** `POST`
- **Description:** Executes a command.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/execute/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "command": "ls -l"
}'
```

---

## Health

### Health Check

- **Endpoint:** `/health`
- **Method:** `GET`
- **Description:** Checks the health of the application.

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

---

## Monitoring

### Get Metrics

- **Endpoint:** `/metrics`
- **Method:** `GET`
- **Description:** Retrieves Prometheus metrics.

**Curl Example:**
```bash
curl -X GET "http://localhost:8000/metrics"
```

---

## RAG (Retrieval-Augmented Generation)

### Query RAG Model

- **Endpoint:** `/rag/query`
- **Method:** `POST`
- **Description:** Queries the RAG model.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/rag/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "query": "What is the capital of France?"
}'
```

### Query RAG Model (Streaming)

- **Endpoint:** `/rag/query_stream`
- **Method:** `POST`
- **Description:** Queries the RAG model and gets a streaming response.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/rag/query_stream" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "query": "What is the capital of France?"
}'
```

---

## Training

### Train Model

- **Endpoint:** `/training/`
- **Method:** `POST`
- **Description:** Trains the model with the provided data.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/training/" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "data": "Some training data."
}'
```

### Train Model from File

- **Endpoint:** `/training/file`
- **Method:** `POST`
- **Description:** Trains the model from a file.
- **Payload:** `multipart/form-data`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/training/file" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-F "file=@/path/to/your/training_file.txt"
```

---

## Voice

### Speech-to-Text

- **Endpoint:** `/voice/stt`
- **Method:** `POST`
- **Description:** Converts speech to text.
- **Payload:** `multipart/form-data`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/voice/stt" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-F "file=@/path/to/your/audio.wav"
```

### Text-to-Speech

- **Endpoint:** `/voice/tts`
- **Method:** `POST`
- **Description:** Converts text to speech.
- **Payload:** `application/json`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/voice/tts" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-d '{
  "text": "Hello, this is a test.",
  "language": "en"
}'
```

---

## Voice Agent

### Interact with Voice Agent

- **Endpoint:** `/voice-agent/`
- **Method:** `POST`
- **Description:** Interacts with the voice agent.
- **Payload:** `multipart/form-data`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/voice-agent/" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-F "file=@/path/to/your/audio.wav"
```

### Interact with Voice Agent (Streaming)

- **Endpoint:** `/voice-agent/stream`
- **Method:** `POST`
- **Description:** Interacts with the voice agent and gets a streaming response.
- **Payload:** `multipart/form-data`

**Curl Example:**
```bash
curl -X POST "http://localhost:8000/voice-agent/stream" \
-H "Authorization: Bearer YOUR_JWT_TOKEN" \
-F "file=@/path/to/your/audio.wav"
```

```