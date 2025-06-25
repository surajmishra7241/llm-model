from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rag_upload():
    response = client.post(
        "/api/v1/rag/upload",
        files={"file": ("test.pdf", b"test content", "application/pdf")},
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert "document_id" in response.json()

def test_rag_query():
    response = client.post(
        "/api/v1/rag/query",
        json={"query": "test query"},
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()