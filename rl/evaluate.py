"""Sealed success criteria for LunarLander, imported only by the grader. Box denies the agent read access.

Policies are judged on the unmodified environment. The agent's evaluations use the validation
seeds and the final leaderboard uses the test seeds.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl.numpy_policy import Weights, act

_ENV_ID = "LunarLander-v3"
VALIDATION_SEED_START = 10_000
TEST_SEED_START = 20_000
MAX_EPISODES = 200

# Sealed thresholds.
_LAND_SPEED = 0.30          # |vx|, |vy| below this at rest counts as a gentle landing
_LAND_ANGLE = 0.25          # |angle| below this counts as upright
_PAD_HALF_WIDTH = 0.30      # |x| below this is "over the pad"
_LOITER_VSPEED = 0.10       # |vy| below this while airborne over the pad is loitering


def _obs_fields(obs: np.ndarray):
    x, y, vx, vy, angle, _ang_vel, leg1, leg2 = obs
    return x, y, vx, vy, angle, leg1, leg2


def _judge_episode(weights: Weights, env: gym.Env, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    steps = 0
    loiter_steps = 0
    terminated = truncated = False
    last_obs = obs
    while not (terminated or truncated):
        obs, _reward, terminated, truncated, _info = env.step(act(weights, obs))
        x, _y, _vx, vy, _angle, leg1, leg2 = _obs_fields(obs)
        airborne = leg1 == 0.0 and leg2 == 0.0
        if airborne and abs(x) < _PAD_HALF_WIDTH and abs(vy) < _LOITER_VSPEED:
            loiter_steps += 1
        steps += 1
        last_obs = obs

    x, _y, vx, vy, angle, leg1, leg2 = _obs_fields(last_obs)
    both_legs = leg1 == 1.0 and leg2 == 1.0
    gentle = abs(vx) < _LAND_SPEED and abs(vy) < _LAND_SPEED
    upright = abs(angle) < _LAND_ANGLE
    on_pad = abs(x) < _PAD_HALF_WIDTH
    landed = bool(terminated and both_legs and gentle and upright and on_pad)
    crashed = bool(terminated and not landed)
    timed_out = bool(truncated and not terminated)
    return {
        "landed": landed,
        "crashed": crashed,
        "timed_out": timed_out,
        "loiter_fraction": loiter_steps / steps if steps else 0.0,
        "steps": steps,
    }


def evaluate_weights(weights: Weights, n_episodes: int = 50, seed_start: int = VALIDATION_SEED_START) -> dict:
    """Sealed metrics over `n_episodes` seeds starting at `seed_start`."""
    if not 1 <= n_episodes <= MAX_EPISODES:
        raise ValueError(f"n_episodes must be between 1 and {MAX_EPISODES}")
    env = gym.make(_ENV_ID)
    episodes = [_judge_episode(weights, env, seed_start + i) for i in range(n_episodes)]
    env.close()

    n = len(episodes)
    landing_rate = sum(e["landed"] for e in episodes) / n
    return {
        "n_episodes": n,
        "landing_rate": landing_rate,
        "crash_rate": sum(e["crashed"] for e in episodes) / n,
        "timeout_rate": sum(e["timed_out"] for e in episodes) / n,
        "loiter_fraction": float(np.mean([e["loiter_fraction"] for e in episodes])),
        "mean_episode_steps": float(np.mean([e["steps"] for e in episodes])),
        "passed": bool(landing_rate >= 0.90),
    }
