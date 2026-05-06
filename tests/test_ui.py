from fastapi.testclient import TestClient

from ai_ks.main import app


def test_home_serves_chat_ui() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Agent conversation" in response.text
    assert 'id="chat-form"' in response.text
    assert 'data-max-query-chars="2000"' in response.text
    assert 'data-api-route="/agent/stream"' in response.text


def test_static_assets_are_served() -> None:
    client = TestClient(app)

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "fetch(\"/health\")" in response.text
