import io

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from rl.numpy_policy import SHAPES, WEIGHTS_FILE, act, export_weights, load_weights, logits


def _rollout_observations(model: PPO, n_obs: int, seed: int = 0) -> np.ndarray:
    """Real rollout observations; 30% random actions widen the states covered."""
    env = gym.make("LunarLander-v3")
    rng = np.random.default_rng(seed)
    observations = []
    obs, _ = env.reset(seed=seed)
    episode = 0
    while len(observations) < n_obs:
        observations.append(obs)
        if rng.random() < 0.3:
            action = int(rng.integers(4))
        else:
            action = int(model.predict(obs, deterministic=True)[0])
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
    env.close()
    return np.array(observations, dtype=np.float32)


def _npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _valid_arrays() -> dict:
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in SHAPES.items()}


def test_numpy_actions_match_sb3_on_trained_policy(trained_run):
    model = PPO.load(trained_run / "policy")
    weights = load_weights(trained_run / WEIGHTS_FILE)
    observations = _rollout_observations(model, n_obs=5_000)

    sb3_actions, _ = model.predict(observations, deterministic=True)
    numpy_actions = np.argmax(logits(weights, observations), axis=-1)

    assert len(observations) >= 5_000
    assert len(set(sb3_actions.tolist())) > 1, "policy should use more than one action"
    np.testing.assert_array_equal(numpy_actions, sb3_actions)
    assert all(act(weights, o) == a for o, a in zip(observations[:200], sb3_actions[:200], strict=True))


def test_numpy_actions_match_sb3_on_untrained_policy(tmp_path):
    model = PPO("MlpPolicy", gym.make("LunarLander-v3"), seed=1)
    export_weights(model, tmp_path / WEIGHTS_FILE)
    weights = load_weights((tmp_path / WEIGHTS_FILE).read_bytes())
    observations = _rollout_observations(model, n_obs=2_000, seed=1)

    sb3_actions, _ = model.predict(observations, deterministic=True)
    np.testing.assert_array_equal(np.argmax(logits(weights, observations), axis=-1), sb3_actions)


def test_load_accepts_valid_weights():
    weights = load_weights(_npz_bytes(**_valid_arrays()))
    assert {name: w.shape for name, w in weights.items()} == SHAPES


def test_load_rejects_pickled_object_array():
    arrays = _valid_arrays()
    arrays["w0"] = np.array([{"not": "weights"}] * 64 * 8, dtype=object).reshape(64, 8)
    with pytest.raises(ValueError):
        load_weights(_npz_bytes(**arrays))


def test_load_rejects_wrong_shape():
    arrays = _valid_arrays()
    arrays["w1"] = np.zeros((64, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        load_weights(_npz_bytes(**arrays))


def test_load_rejects_extra_or_missing_keys():
    with pytest.raises(ValueError, match="exactly"):
        load_weights(_npz_bytes(**_valid_arrays(), extra=np.zeros(3)))
    arrays = _valid_arrays()
    del arrays["b2"]
    with pytest.raises(ValueError, match="exactly"):
        load_weights(_npz_bytes(**arrays))


def test_load_rejects_non_finite_and_non_float():
    arrays = _valid_arrays()
    arrays["b0"][3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        load_weights(_npz_bytes(**arrays))
    arrays = _valid_arrays()
    arrays["b0"] = np.zeros(64, dtype=np.int64)
    with pytest.raises(ValueError, match="dtype"):
        load_weights(_npz_bytes(**arrays))


def test_load_rejects_non_zip():
    with pytest.raises(ValueError, match="npz"):
        load_weights(b"PK not really a zip")
