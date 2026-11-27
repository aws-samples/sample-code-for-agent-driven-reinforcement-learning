import base64
import http.client
import json
import subprocess
import sys
import threading

import pytest

from rl.evaluate import TEST_SEED_START, VALIDATION_SEED_START, evaluate_weights
from rl.grader import make_server
from rl.grader_client import GraderError, evaluate, health
from rl.numpy_policy import WEIGHTS_FILE, load_weights


@pytest.fixture(scope="module")
def grader():
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()
    server.server_close()


def _url(address) -> str:
    return f"http://{address[0]}:{address[1]}"


def _post(address, path: str, body: bytes, headers: dict | None = None):
    conn = http.client.HTTPConnection(*address, timeout=30)
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    result = response.status, json.loads(response.read())
    conn.close()
    return result


def _body(trained_run, n_episodes=5) -> bytes:
    raw = (trained_run / WEIGHTS_FILE).read_bytes()
    return json.dumps({"weights_b64": base64.b64encode(raw).decode(), "n_episodes": n_episodes}).encode()


def test_health(grader):
    assert health(_url(grader))
    assert not health("http://127.0.0.1:9")


def test_evaluate_and_final_use_separate_seed_ranges(grader, trained_run):
    weights = load_weights(trained_run / WEIGHTS_FILE)
    validation = evaluate(trained_run, n_episodes=5, url=_url(grader))
    final = evaluate(trained_run, n_episodes=5, endpoint="final", url=_url(grader))

    assert validation == evaluate_weights(weights, 5, VALIDATION_SEED_START)
    assert final == evaluate_weights(weights, 5, TEST_SEED_START)
    assert set(validation) >= {"landing_rate", "crash_rate", "timeout_rate", "loiter_fraction", "passed"}


def test_concurrent_requests(grader, trained_run):
    results = [None] * 3

    def call(i):
        results[i] = evaluate(trained_run, n_episodes=3, url=_url(grader))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results[0] == results[1] == results[2]


def test_bad_requests(grader, trained_run):
    assert _post(grader, "/evaluate", b"not json")[0] == 400
    assert _post(grader, "/evaluate", json.dumps({"run_dir": "artifacts/iter_1"}).encode())[0] == 400
    assert _post(grader, "/evaluate", json.dumps({"weights_b64": "@@@"}).encode())[0] == 400
    junk = base64.b64encode(b"not an npz").decode()
    assert _post(grader, "/evaluate", json.dumps({"weights_b64": junk}).encode())[0] == 400
    assert _post(grader, "/evaluate", _body(trained_run, n_episodes=0))[0] == 400
    assert _post(grader, "/evaluate", _body(trained_run, n_episodes=201))[0] == 400
    assert _post(grader, "/evaluate", _body(trained_run, n_episodes=True))[0] == 400


def test_unknown_path_and_oversize_body(grader):
    assert _post(grader, "/run", b"{}")[0] == 404
    status, body = _post(grader, "/evaluate", b"{}", headers={"Content-Length": "2000000"})
    assert status == 413
    conn = http.client.HTTPConnection(*grader, timeout=5)
    conn.request("GET", "/evaluate")
    assert conn.getresponse().status == 404
    conn.close()


def test_client_reports_missing_weights(tmp_path, grader):
    with pytest.raises(GraderError, match=WEIGHTS_FILE):
        evaluate(tmp_path, url=_url(grader))


def test_grader_process_never_imports_torch(trained_run):
    script = (
        "import sys\n"
        "from rl.evaluate import evaluate_weights\n"
        "from rl.grader import make_server\n"
        "from rl.numpy_policy import load_weights\n"
        f"evaluate_weights(load_weights({str(trained_run / WEIGHTS_FILE)!r}), 2)\n"
        "loaded = [m for m in ('torch', 'stable_baselines3', 'cloudpickle') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
