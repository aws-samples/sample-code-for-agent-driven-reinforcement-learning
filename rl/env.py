"""LunarLander with the reward replaced by an agent-authored function, clamped to ±1000."""

from __future__ import annotations

import math
from collections.abc import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

ENV_ID = "LunarLander-v3"
_REWARD_CLAMP = 1000.0

RewardFn = Callable[..., float]


class CustomRewardEnv(gym.Wrapper):
    """Wrap LunarLander and replace its reward with `reward_fn`."""

    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, _base_reward, terminated, truncated, info = self.env.step(action)
        reward = self._safe_reward(self._prev_obs, action, next_obs, terminated, truncated, info)
        self._prev_obs = next_obs
        return next_obs, reward, terminated, truncated, info

    def _safe_reward(self, obs, action, next_obs, terminated, truncated, info) -> float:
        try:
            value = float(self._reward_fn(obs, action, next_obs, terminated, truncated, info))
        except Exception:
            return -_REWARD_CLAMP
        if not math.isfinite(value):
            return -_REWARD_CLAMP
        return max(-_REWARD_CLAMP, min(_REWARD_CLAMP, value))


def make_env(reward_fn: RewardFn, seed: int | None = None) -> Callable[[], gym.Env]:
    """Return a thunk that builds one wrapped env (for SB3 vectorised envs)."""

    def _thunk() -> gym.Env:
        base = gym.make(ENV_ID)
        env = Monitor(CustomRewardEnv(base, reward_fn))
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk
