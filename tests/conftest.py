import pathlib

import pytest

from rl.train import train

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("run")
    train(str(REPO / "rl/rewards/examples/landing.py"), steps=20_000, seed=0, out_dir=str(out), n_envs=4)
    return out
