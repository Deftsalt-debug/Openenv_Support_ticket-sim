"""Bad requests should leave an episode usable, over HTTP and WebSocket."""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        client.post("/reset", json={"task_id": "task_1", "seed": 42})
        yield client


@pytest.mark.parametrize("body", [[], "hello", None, {"priority": "oops"}, {"action": []}, {"action": None}])
def test_invalid_step_does_not_advance_episode(client, body):
    # Use explicit JSON text so None is sent as null, rather than an empty body.
    import json

    response = client.post("/step", content=json.dumps(body), headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert "traceback" not in response.json()
    assert client.get("/state").json()["step_count"] == 0
    assert client.post("/step", json={"action": {"priority": "P1_HIGH"}}).status_code == 200


@pytest.mark.parametrize("body", ["{", "[]", "null", '{"seed": null}', '{"task_id": "task_99"}'])
def test_invalid_reset_preserves_episode(client, body):
    before = client.get("/state").json()
    response = client.post("/reset", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert client.get("/state").json() == before


def test_empty_reset_and_direct_action_are_still_supported(client):
    assert client.post("/reset").status_code == 200
    response = client.post("/step", json={"priority": "P1_HIGH"})
    assert response.status_code == 200
    assert response.json()["observation"]["reward"] == response.json()["reward"]


def test_websocket_recovers_from_invalid_messages(client):
    with client.websocket_connect("/ws") as socket:
        socket.send_json([])
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"type": "reset", "seed": "42"})
        assert socket.receive_json()["type"] == "reset_result"
        socket.send_json({"type": "step", "action": []})
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"type": "state"})
        assert socket.receive_json()["state"]["step_count"] == 0
        socket.send_json({"type": "step", "action": {"priority": "P1_HIGH"}})
        assert socket.receive_json()["type"] == "step_result"
