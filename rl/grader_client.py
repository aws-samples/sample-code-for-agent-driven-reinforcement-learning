"""HTTP client for the grader (`rl.grader`)."""

from __future__ import annotations

import base64
import http.client
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from rl.numpy_policy import WEIGHTS_FILE

DEFAULT_URL = "http://127.0.0.1:8899"


class GraderError(RuntimeError):
    pass


def grader_url() -> str:
    return os.environ.get("GRADER_URL", DEFAULT_URL).rstrip("/")


def health(url: str | None = None, timeout: float = 2.0) -> bool:
    """True if a grader answers `GET /health` at `url`."""
    try:
        with urllib.request.urlopen(f"{url or grader_url()}/health", timeout=timeout) as response:
            return json.loads(response.read()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def evaluate(
    run_dir: str | os.PathLike,
    n_episodes: int = 50,
    endpoint: str = "evaluate",
    url: str | None = None,
    timeout: float = 600.0,
) -> dict:
    """Send `<run_dir>/policy_weights.npz` to `endpoint` ("evaluate" or "final") and return the metrics."""
    weights_path = pathlib.Path(run_dir) / WEIGHTS_FILE
    if not weights_path.is_file():
        raise GraderError(f"no {WEIGHTS_FILE} in {run_dir}; train a policy there first")
    body = json.dumps(
        {"weights_b64": base64.b64encode(weights_path.read_bytes()).decode("ascii"), "n_episodes": n_episodes}
    ).encode("utf-8")
    target = f"{url or grader_url()}/{endpoint}"
    try:
        status, raw = _post(target, body, timeout)
    except OSError as exc:
        raise GraderError(f"grader not reachable at {url or grader_url()}: {exc}") from exc
    if status != 200:
        raise GraderError(f"grader returned {status}: {_error_message(raw)}")
    return json.loads(raw)


def _post(target: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json"}
    proxy = os.environ.get("HTTP_PROXY") if os.environ.get("AUTORL_GRADER_VIA_PROXY") == "1" else None
    if proxy:
        # Inside Strands Box, NO_PROXY covers 127.0.0.1, but the sandbox only lets traffic out through
        # the box's egress gateway, so send the request to the gateway ourselves.
        parsed = urllib.parse.urlparse(proxy)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
        try:
            connection.request("POST", target, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()
    request = urllib.request.Request(target, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _error_message(raw: bytes) -> str:
    try:
        return json.loads(raw).get("error", raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return raw.decode("utf-8", "replace")


def ensure_grader(url: str | None = None, wait_seconds: float = 30.0) -> bool:
    """Start a local grader if none answers at `url`; True if one was started.

    It outlives the caller (own session) and logs to `autorl-grader-<port>.log` in the temp directory.
    """
    url = url or grader_url()
    if health(url):
        return False
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise GraderError(f"no grader at {url}, and only a local grader can be started from here")
    port = parsed.port or 80
    env = {**os.environ, "GRADER_HOST": "127.0.0.1", "GRADER_PORT": str(port)}
    log_path = pathlib.Path(tempfile.gettempdir()) / f"autorl-grader-{port}.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "rl.grader"], env=env, stdout=log, stderr=log, start_new_session=True
        )
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if health(url):
            return True
        if process.poll() is not None:
            lines = log_path.read_text(errors="replace").strip().splitlines()
            raise GraderError(
                f"the grader exited at startup ({lines[-1] if lines else 'no output'}); "
                f"is something else using port {port}? Log: {log_path}"
            )
        time.sleep(0.2)
    raise GraderError(f"started a grader but it did not answer at {url} within {wait_seconds:.0f}s")
