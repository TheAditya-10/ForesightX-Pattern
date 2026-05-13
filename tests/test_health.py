from __future__ import annotations

from fastapi.testclient import TestClient

from foresightx_pattern.app.main import create_app


def test_health_endpoint_returns_ok():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") in {"ok", "healthy", "up", "running", "ready"}
