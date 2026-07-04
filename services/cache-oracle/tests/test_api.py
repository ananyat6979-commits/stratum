"""
Tests for the cache-oracle HTTP API.

Uses FastAPI's TestClient, which drives the app in-process without a
real TCP listener -- consistent with the project's existing preference
for oneshot()-style testing over network-bound integration tests
(see stratum-gateway's ingress.rs tests, same philosophy).
"""

import pytest
from fastapi.testclient import TestClient

from stratum_oracle.api import app, collector


@pytest.fixture(autouse=True)
def reset_collector():
    """Ensure each test starts with a clean collector state."""
    collector._worker_addresses.clear()
    collector._predictors.clear()
    collector._metrics.clear()
    yield


def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_register_worker_succeeds():
    with TestClient(app) as client:
        response = client.post(
            "/workers/register",
            json={"worker_id": "worker-0", "address": "http://127.0.0.1:11434", "backend_type": "ollama"},
        )
        assert response.status_code == 200
        assert response.json()["worker_id"] == "worker-0"


def test_signals_for_unscraped_worker_returns_neutral_zero_observations():
    with TestClient(app) as client:
        client.post(
            "/workers/register",
            json={"worker_id": "worker-0", "address": "http://127.0.0.1:11434", "backend_type": "ollama"},
        )
        response = client.get("/signals")
        assert response.status_code == 200
        data = response.json()
        assert len(data["workers"]) == 1
        worker = data["workers"][0]
        assert worker["worker_id"] == "worker-0"
        assert worker["n_observations"] == 0
        assert worker["cache_hit_prob_is_real"] is False
        assert worker["latency_sla_signals_are_real"] is False


def test_signals_response_explicitly_labels_placeholder_fields():
    """
    The response must be honest: cache_hit_prob, predicted_latency_ms,
    and sla_affinity are not yet implemented, and the response schema
    must say so via the _is_real flags, not silently return plausible-
    looking numbers with no indication they're placeholders.
    """
    with TestClient(app) as client:
        client.post(
            "/workers/register",
            json={"worker_id": "worker-0", "address": "http://127.0.0.1:11434", "backend_type": "ollama"},
        )
        response = client.get("/signals")
        worker = response.json()["workers"][0]
        assert worker["cache_hit_prob_is_real"] is False
        assert worker["latency_sla_signals_are_real"] is False


def test_deregister_worker_removes_from_signals():
    with TestClient(app) as client:
        client.post(
            "/workers/register",
            json={"worker_id": "worker-0", "address": "http://127.0.0.1:11434", "backend_type": "ollama"},
        )
        client.delete("/workers/worker-0")
        response = client.get("/signals")
        assert len(response.json()["workers"]) == 0


def test_signals_with_no_registered_workers_returns_empty_list():
    with TestClient(app) as client:
        response = client.get("/signals")
        assert response.json()["workers"] == []