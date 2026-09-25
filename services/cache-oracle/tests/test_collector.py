"""
Tests for collector.py.

This file did not exist before this fix. Its absence is exactly why
the bug it tests for shipped undetected: nothing exercised
_fetch_ollama_utilization's or _scrape_worker's failure paths.

The bug: _fetch_ollama_utilization used to catch every exception
(connection refused, timeout, malformed JSON) and silently return
0.0. That 0.0 flowed back through _fetch_utilization into
_scrape_worker as a clean, successful return value, no exception
ever reached _scrape_worker's own except block, whose comment
promises "record as high pressure (conservative)" on failure. A
broken or unreachable Ollama worker was therefore recorded as
scrape_success=True with kv_utilization=0.0: indistinguishable from a
genuinely idle, healthy worker, and read by KvPressurePredictor and
the router as the safest possible place to send traffic.

The fix makes _fetch_ollama_utilization (and _fetch_utilization)
return None specifically on a failed scrape, and _scrape_worker checks
for None explicitly rather than relying on an exception to propagate.
These tests exercise that distinction directly, using
httpx.MockTransport to simulate real connection failures without a
live server.
"""

import asyncio

import httpx
import pytest

from stratum_oracle.collector import MetricsCollector


def run(coro):
    """Run an async test body without requiring pytest-asyncio, which
    is not a declared dependency of this project."""
    return asyncio.run(coro)


def _client_with_transport(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


def test_connection_failure_is_recorded_as_high_pressure_not_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    assert metrics is not None, "a metrics entry must be recorded even on the first-ever failed scrape"
    assert metrics.scrape_success is False, (
        "a connection failure must be marked as a failed scrape"
    )
    assert metrics.kv_utilization == 1.0, (
        "a failed scrape must be recorded as high pressure (1.0), not a "
        "clean 0.0 that is indistinguishable from a genuinely idle worker"
    )


def test_timeout_is_recorded_as_high_pressure_not_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    assert metrics.scrape_success is False
    assert metrics.kv_utilization == 1.0


def test_malformed_json_is_recorded_as_high_pressure_not_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all {{{")

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    assert metrics.scrape_success is False
    assert metrics.kv_utilization == 1.0


def test_http_500_is_recorded_as_high_pressure_not_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal server error")

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    assert metrics.scrape_success is False
    assert metrics.kv_utilization == 1.0


def test_genuinely_idle_worker_still_reports_a_real_zero():
    """
    The critical distinction this fix must preserve in both
    directions: a REAL idle worker (successful scrape, zero models
    loaded) must still report scrape_success=True and kv_utilization
    0.0. The fix must not turn every 0.0 into a failure; it must only
    stop treating failures as a 0.0.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    assert metrics.scrape_success is True
    assert metrics.kv_utilization == 0.0


def test_loaded_model_reports_real_nonzero_utilization():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3", "size_vram": 4 * 1024 * 1024 * 1024}]},
        )

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    assert metrics.scrape_success is True
    # 4GB / 8GB conservative max = 0.5
    assert metrics.kv_utilization == pytest.approx(0.5)


def test_multiple_loaded_models_sum_vram_not_just_the_first():
    """
    Regression test for the stale comment bug found alongside the main
    fix: the code already summed size_vram across all loaded models,
    but the comment claimed it only used the first model. This test
    pins down the real, correct behavior (sum across all models) so a
    future edit cannot silently narrow it to match the old, incorrect
    comment.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3", "size_vram": 2 * 1024 * 1024 * 1024},
                    {"name": "mistral", "size_vram": 2 * 1024 * 1024 * 1024},
                ]
            },
        )

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
        return collector.get_metrics("worker-0")

    metrics = run(body())

    # (2GB + 2GB) / 8GB conservative max = 0.5, not 2GB / 8GB = 0.25
    # (which is what "only the first model" would have produced)
    assert metrics.kv_utilization == pytest.approx(0.5)


def test_failure_followed_by_recovery_reports_real_reading_again():
    """
    A worker that fails once and then recovers must go back to
    reporting real, honest readings, the high-pressure fallback must
    not stick permanently after a single failure.
    """
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"models": []})

    async def body():
        collector = MetricsCollector()
        collector.register_worker("worker-0", "http://fake-ollama:11434", "ollama")
        async with _client_with_transport(httpx.MockTransport(handler)) as client:
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
            first = collector.get_metrics("worker-0")
            await collector._scrape_worker(
                client, "worker-0", "http://fake-ollama:11434", "ollama"
            )
            second = collector.get_metrics("worker-0")
        return first, second

    first, second = run(body())

    assert first.scrape_success is False
    assert first.kv_utilization == 1.0
    assert second.scrape_success is True
    assert second.kv_utilization == 0.0