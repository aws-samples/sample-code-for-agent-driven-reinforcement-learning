"""Sealed grader service, the only importer of rl.evaluate. Runs on the host, outside the agent's box.

    GET  /health    -> {"status": "ok"}
    POST /evaluate  {"weights_b64": ..., "n_episodes": N} -> metrics on the validation seeds
    POST /final     same body -> metrics on the test seeds (leaderboard only)

    python -m rl.grader            # GRADER_HOST (127.0.0.1) and GRADER_PORT (8899) override
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rl.evaluate import TEST_SEED_START, VALIDATION_SEED_START, evaluate_weights
from rl.numpy_policy import load_weights

_MAX_BODY_BYTES = 1_000_000
_SEED_STARTS = {"/evaluate": VALIDATION_SEED_START, "/final": TEST_SEED_START}


class _BadRequest(Exception):
    pass


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        seed_start = _SEED_STARTS.get(self.path)
        if seed_start is None:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"error": "invalid Content-Length"})
            return
        if length > _MAX_BODY_BYTES:
            self.close_connection = True
            self._send(413, {"error": f"request body over {_MAX_BODY_BYTES} bytes"})
            return
        try:
            weights, n_episodes = _parse(self.rfile.read(length))
            metrics = evaluate_weights(weights, n_episodes, seed_start)
        except (_BadRequest, ValueError) as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # never send a traceback
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, metrics)

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def _parse(body: bytes):
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as exc:
        raise _BadRequest("body must be JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("weights_b64"), str):
        raise _BadRequest('body must be {"weights_b64": "<base64 of policy_weights.npz>", "n_episodes": N}')
    try:
        raw = base64.b64decode(payload["weights_b64"], validate=True)
    except binascii.Error as exc:
        raise _BadRequest("weights_b64 is not valid base64") from exc
    n_episodes = payload.get("n_episodes", 50)
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool):
        raise _BadRequest("n_episodes must be an integer")
    return load_weights(raw), n_episodes


def make_server(host: str = "127.0.0.1", port: int = 8899) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def main() -> None:
    host = os.environ.get("GRADER_HOST", "127.0.0.1")
    port = int(os.environ.get("GRADER_PORT", "8899"))
    server = make_server(host, port)
    print(f"grader listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
