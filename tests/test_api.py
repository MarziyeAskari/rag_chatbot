from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_returns_200():
    r = client.get("/")
    assert r.status_code == 200

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200

