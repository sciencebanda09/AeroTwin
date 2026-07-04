from fastapi.testclient import TestClient
from pipeline import demo_data
from src.dataset.loader import FEATURES
from src.api.server import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_update_endpoint_returns_confidence() -> None:
    payload = {name: float(demo_data(1, 1).iloc[0][name]) for name in FEATURES}
    with TestClient(app) as client:
        response = client.post("/v1/engines/test-engine/update", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["Confidence"] >= 0
        assert "RULCyclesLower" in body
