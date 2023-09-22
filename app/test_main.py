from fastapi.testclient import TestClient

from app.model import app

client = TestClient(app)

def test_get_threshold():
    response = client.post("/get_threshold")
    assert response.status_code == 200
    assert response.json() == {"data": 0.5}


