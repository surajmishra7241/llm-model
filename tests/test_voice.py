from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_speech_to_text():
    response = client.post(
        "/api/v1/voice/stt",
        files={"audio_file": ("test.wav", b"test audio", "audio/wav")},
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert "text" in response.json()

def test_text_to_speech():
    response = client.post(
        "/api/v1/voice/tts",
        json={"text": "test text"},
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"