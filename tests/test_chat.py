from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello",
            "model": "llama3",
            "system_prompt": "You are a helpful assistant"
        },
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert "message" in response.json()