"""
Minimal stub worker for the semantic_vs_round_robin benchmark.

WHY THIS EXISTS
================
Both prior attempts at this benchmark used a permanently unreachable
worker (http://127.0.0.1:0 or :1) as the dispatch target, matching the
pattern used throughout this project's test suite for deterministic
dispatch-failure tests. That pattern is correct for unit tests, which
only need a handful of requests. It is wrong for a 120-second benchmark
run against SemanticRouter's WorkerRegistry specifically: every
dispatch failure counts toward WorkerRegistry's 10-consecutive-failure
Unavailable threshold, and a permanently unreachable worker accumulates
exactly 10 failures roughly two minutes into any real run, at which
point routable_workers() returns empty and every subsequent request
fails closed with 503 instead of measuring routing overhead.

Adding a second permanently-unreachable worker (STRATUM_WORKER_1_URL)
only doubled the runway before both workers independently crossed the
same threshold. It did not remove the ceiling, because
WorkerRegistry has no recovery path once a worker is Unavailable,
only a successful dispatch (record_success) resets the counter, and a
genuinely dead worker never produces one.

This script is that success path. It's a real, listening HTTP server
that answers every POST to /api/generate with a fast, valid-looking
Ollama-compatible 200 response, so record_success() actually fires on
(nearly) every request, no worker ever accumulates enough consecutive
failures to go Unavailable, and the benchmark measures what it's
supposed to measure: gateway + routing overhead, not circuit-breaker
behavior under total, permanent worker failure.

USAGE
=====
    uv run python stub_worker.py --port 9000

Run one instance per port needed (9000 for worker-0, 9001 for
worker-1, or point both STRATUM_WORKER_0_URL and STRATUM_WORKER_1_URL
at the same instance, either is fine, this server has no state that
would make sharing it between two "workers" meaningfully different
from having two separate ones for this benchmark's purposes).
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubOllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(content_length)  # drain the request body, unused

        body = json.dumps(
            {
                "model": "phi3:mini",
                "response": "4",
                "done": True,
            }
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        # Silence per-request console logging, the gateway's own
        # tracing output is the log that matters for this benchmark,
        # this server's console noise would just bury it.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal stub Ollama-compatible worker")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), StubOllamaHandler)
    print(f"stub_worker listening on http://127.0.0.1:{args.port}")
    print("answers every POST with a fast, fixed 200 response")
    server.serve_forever()


if __name__ == "__main__":
    main()