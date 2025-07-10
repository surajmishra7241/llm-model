from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_current_user
from app.services.agent_service import AgentService
from app.services.voice_agent_service import VoiceAgentService
from app.services.voice_service import VoiceService
from app.models.agent_model import Agent
import pytest

# Mock user dependency
async def override_get_current_user():
    return {"user_id": 1, "username": "testuser"}

app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

@pytest.fixture
def mock_agent_service(mocker):
    mock = mocker.AsyncMock(spec=AgentService)
    mock.get_agent.return_value = Agent(
        id=1,
        name="Test Voice Agent",
        description="A test agent for voice chat",
        system_prompt="You are a helpful voice assistant.",
        model="test-model",
        rag_enabled=False,
        owner_id=1
    )
    return mock

@pytest.fixture
def mock_voice_service(mocker):
    mock = mocker.AsyncMock(spec=VoiceService)
    mock.speech_to_text.return_value = "Hello, this is a test."
    mock.text_to_speech.return_value = b"mock_audio_bytes"
    return mock

@pytest.fixture
def mock_voice_agent_service(mocker):
    mock = mocker.AsyncMock(spec=VoiceAgentService)
    mock.voice_chat.return_value = "This is a mock response from the agent."
    return mock

def test_voice_chat_with_agent(
    mock_agent_service,
    mock_voice_service,
    mock_voice_agent_service
):
    app.dependency_overrides[AgentService] = lambda: mock_agent_service
    app.dependency_overrides[VoiceService] = lambda: mock_voice_service
    app.dependency_overrides[VoiceAgentService] = lambda: mock_voice_agent_service

    response = client.post(
        "/api/v1/voice-agent/voice-chat/1",
        files={"audio_file": ("test.wav", b"dummy audio data", "audio/wav")}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"mock_audio_bytes"

    # Reset overrides
    app.dependency_overrides = {}
    app.dependency_overrides[get_current_user] = override_get_current_user

