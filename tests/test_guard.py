import pathlib

import pytest

from rl.guard import RewardGuardError, load_reward, validate

SIGNATURE = "def reward(obs, action, next_obs, terminated, truncated, info):\n    "
REPO = pathlib.Path(__file__).resolve().parent.parent


def _source(body: str) -> str:
    return SIGNATURE + body


@pytest.mark.parametrize(
    "path",
    sorted((REPO / "rl" / "rewards" / "examples").glob("*.py")) + sorted((REPO / "results").glob("*/*/rewards/*.py")),
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_existing_rewards_pass(path):
    validate(path.read_text())


@pytest.mark.parametrize(
    "body",
    [
        "return math.sqrt(abs(obs[0])) + math.pi",
        "return float(np.clip(np.tanh(obs[0]), -1.0, 1.0))",
        "return float(np.linalg.norm(np.asarray(next_obs[:2])))",
        "return -math.inf if terminated else max(0.0, np.inf - 1.0)",
    ],
)
def test_allowed(body):
    fn = load_reward(_source(body))
    assert isinstance(fn([0.5] * 8, 0, [0.5] * 8, False, False, {}), float)


@pytest.mark.parametrize(
    "body",
    [
        "np.save('/tmp/x', obs); return 0.0",
        "return float(np.load('/tmp/x.npy', allow_pickle=True))",
        "np.lib.format.open_memmap('/tmp/y', mode='w+', shape=(1,)); return 0.0",
        "np.ctypeslib.load_library('libc', '/usr/lib'); return 0.0",
        "np.zeros(1).tofile('/tmp/z'); return 0.0",
        "obs.tofile('/tmp/z'); return 0.0",
        "f = np.save; return 0.0",
        "return math.floor(obs[0])",
        "return (lambda: 0.0)()",
        "return obs.__class__",
        "import os; return 0.0",
        "return eval('0.0')",
    ],
)
def test_rejected(body):
    with pytest.raises(RewardGuardError):
        validate(_source(body))
