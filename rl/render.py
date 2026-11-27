"""Record one episode of a policy to a GIF, from its policy_weights.npz (random actions without one)."""

from __future__ import annotations

import argparse
import os
import pathlib

import gymnasium as gym
import numpy as np
from PIL import Image

from rl.env import ENV_ID
from rl.numpy_policy import WEIGHTS_FILE, act, load_weights

_FPS = 50


def record(
    run_dir: str | os.PathLike | None,
    out_path: str,
    seed: int = 10_000,
    max_steps: int = 1000,
    hold_seconds: float = 1.5,
    scale: float = 0.5,
    frame_skip: int = 2,
) -> str:
    """Write a GIF of one episode; `run_dir=None` flies the lander with random actions."""
    weights = load_weights(pathlib.Path(run_dir) / WEIGHTS_FILE) if run_dir else None
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        action = act(weights, obs) if weights is not None else int(rng.integers(4))
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if steps % frame_skip == 0:
            frames.append(env.render())
    frames.append(env.render())
    env.close()

    # Episodes end the moment the lander settles, so hold the last frame long enough to see it.
    frames.extend([frames[-1]] * int(hold_seconds * _FPS / frame_skip))

    height, width = frames[0].shape[:2]
    size = (int(width * scale), int(height * scale))
    images = [Image.fromarray(frame).resize(size) for frame in frames]
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 * frame_skip / _FPS), loop=0)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a policy rollout to a GIF.")
    parser.add_argument("--run-dir", help="Folder with policy_weights.npz (random actions if left out)")
    parser.add_argument("--out", required=True, help="Output .gif")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    print("wrote", record(args.run_dir, args.out, args.seed, args.max_steps))


if __name__ == "__main__":
    main()
