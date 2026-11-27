"""PPO training entrypoint, run by the agent from its shell.

Writes `policy.zip`, `policy_weights.npz` (for the grader) and `train_summary.json` to `--out`.
Never imports rl.evaluate. `AUTORL_MAX_STEPS` caps `--steps`; `AUTORL_TORCH_THREADS` (default 1)
sets torch's thread count.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env import make_env
from rl.guard import load_reward_file
from rl.numpy_policy import WEIGHTS_FILE, export_weights


def train(
    reward_path: str,
    steps: int,
    seed: int,
    out_dir: str,
    n_envs: int = 8,
    gamma: float = 0.999,
    ent_coef: float = 0.01,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
) -> dict:
    max_steps = os.environ.get("AUTORL_MAX_STEPS")
    if max_steps and steps > int(max_steps):
        raise SystemExit(f"--steps {steps} is over this run's limit of {int(max_steps)}")
    torch.set_num_threads(int(os.environ.get("AUTORL_TORCH_THREADS", "1")))
    reward_fn = load_reward_file(reward_path)
    vec_env = DummyVecEnv([make_env(reward_fn, seed=seed + i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=0.98,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        verbose=0,
    )
    model.learn(total_timesteps=steps, progress_bar=False)

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "policy"))
    export_weights(model, out / WEIGHTS_FILE)

    ep_info = list(getattr(model, "ep_info_buffer", []) or [])
    rewards = [e["r"] for e in ep_info]
    summary = {
        "reward_path": reward_path,
        "steps": steps,
        "seed": seed,
        "n_envs": n_envs,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "train_mean_episode_reward": float(sum(rewards) / len(rewards)) if rewards else None,
        "train_episodes_recorded": len(rewards),
        "checkpoint": str(out / "policy.zip"),
        "weights": str(out / WEIGHTS_FILE),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    vec_env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander with a custom reward.")
    parser.add_argument("--reward", required=True, help="Path to the generated reward .py file")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Run output directory, e.g. artifacts/iter_1")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.999, help="PPO discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="PPO entropy coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length per env")
    args = parser.parse_args()

    summary = train(
        args.reward, args.steps, args.seed, args.out, args.n_envs,
        gamma=args.gamma, ent_coef=args.ent_coef, learning_rate=args.lr, n_steps=args.n_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
