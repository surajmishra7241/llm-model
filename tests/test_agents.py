from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_agent():
    response = client.post(
        "/api/v1/agents",
        json={
            "name": "Test Agent",
            "description": "Test Description",
            "model": "llama3",
            "system_prompt": "You are a test agent",
            "is_public": False,
            "tools": [],
            "metadata": {}
        },
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert "id" in response.json()

def test_list_agents():
    response = client.get(
        "/api/v1/agents",
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)