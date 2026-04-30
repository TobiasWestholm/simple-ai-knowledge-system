from fastapi.testclient import TestClient

from ai_ks.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "alive"
    assert payload["models"]["llm_model_id"] == "batiai/gemma4-e2b"
    assert "qdrant" in payload["services"]
    assert "embedding" in payload["services"]
