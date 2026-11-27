"""Train a reward and judge checkpoints through the grader, writing history.json for the gap plot.

Each entry pairs the training reward with the sealed landing rate. Needs a grader running.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env import make_env
from rl.grader_client import evaluate
from rl.guard import load_reward_file
from rl.numpy_policy import WEIGHTS_FILE, export_weights


class _CurveCallback(BaseCallback):
    def __init__(self, eval_every: int, n_eval: int, ckpt_dir: str, history: list):
        super().__init__()
        self.eval_every = eval_every
        self.n_eval = n_eval
        self.ckpt_dir = ckpt_dir
        self.history = history
        self._last = 0

    def _record(self) -> None:
        export_weights(self.model, f"{self.ckpt_dir}/{WEIGHTS_FILE}")
        metrics = evaluate(self.ckpt_dir, self.n_eval)
        ep = list(getattr(self.model, "ep_info_buffer", []) or [])
        train_reward = float(sum(e["r"] for e in ep) / len(ep)) if ep else None
        self.history.append(
            {
                "timesteps": int(self.num_timesteps),
                "train_reward": train_reward,
                "landing_rate": metrics["landing_rate"],
                "loiter_fraction": metrics["loiter_fraction"],
                "crash_rate": metrics["crash_rate"],
            }
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.eval_every:
            self._last = self.num_timesteps
            self._record()
        return True


def run(reward_path, steps, out_dir, eval_every=20_000, n_eval=15, n_envs=8, seed=0) -> list:
    reward_fn = load_reward_file(reward_path)
    vec_env = DummyVecEnv([make_env(reward_fn, seed=seed + i) for i in range(n_envs)])
    model = PPO(
        "MlpPolicy", vec_env, seed=seed, n_steps=1024, batch_size=64, n_epochs=4,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01, verbose=0,
    )
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history: list = []
    model.learn(total_timesteps=steps, callback=_CurveCallback(eval_every, n_eval, str(out), history))
    model.save(str(out / "policy"))
    export_weights(model, out / WEIGHTS_FILE)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    vec_env.close()
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with periodic sealed evaluation, saving a curve.")
    parser.add_argument("--reward", required=True)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--n-eval", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    history = run(args.reward, args.steps, args.out, args.eval_every, args.n_eval, seed=args.seed)
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
