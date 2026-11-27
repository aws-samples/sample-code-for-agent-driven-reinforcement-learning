"""Plots of training reward against the sealed landing rate, and of landing rate per iteration."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colours, readable with red-green colour blindness; lines also differ by marker.
_LAND = "#56B4E9"
_REWARD = "#E69F00"
_AGENTS = [("#56B4E9", "o"), ("#E69F00", "s"), ("#CC79A7", "^")]
_BG = "#0d0d0d"


def _load(path: str) -> list:
    with open(path) as fh:
        return json.load(fh)


def _ema(values: list, alpha: float = 0.4) -> list:
    """Exponential moving average; None counts as 0."""
    out: list = []
    smoothed = None
    for v in values:
        v = 0.0 if v is None else v
        smoothed = v if smoothed is None else alpha * v + (1 - alpha) * smoothed
        out.append(smoothed)
    return out


def _panel(ax, hist: list, title: str) -> None:
    ts = [h["timesteps"] / 1000 for h in hist]
    landing = _ema([h["landing_rate"] for h in hist])
    reward = _ema([h["train_reward"] for h in hist])

    ax.set_title(title, fontsize=15, fontweight="bold", color="w", pad=12)
    ax.plot(ts, landing, color=_LAND, linewidth=3.5, marker="o", label="Sealed landing rate")
    ax.axhline(0.9, color="#888888", linestyle=":", linewidth=2, label="target 0.90")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("training steps (thousands)", fontsize=11, color="#aaaaaa")
    ax.set_ylabel("landing rate", fontsize=12, color=_LAND, fontweight="bold")
    ax.tick_params(colors="#aaaaaa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    twin = ax.twinx()
    twin.plot(ts, reward, color=_REWARD, linewidth=3.5, linestyle="--", marker="s",
              label="Training reward")
    twin.set_ylabel("training reward", fontsize=12, color=_REWARD, fontweight="bold")
    twin.tick_params(colors="#aaaaaa")
    twin.spines["top"].set_visible(False)

    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right",
              facecolor="#1a1a1a", edgecolor="#333333", labelcolor="w", fontsize=10)


def plot(naive_path: str, fixed_path: str, out_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=_BG)
    fig.suptitle("Training reward vs sealed landing rate",
                 fontsize=18, fontweight="bold", color="w")
    for ax in axes:
        ax.set_facecolor(_BG)
    _panel(axes[0], _load(naive_path), "Naive reward: misspecified")
    _panel(axes[1], _load(fixed_path), "Fixed reward: converges")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, facecolor=_BG)
    return out_path


def plot_gap(history_path: str, out_path: str, title: str = "The verification gap") -> str:
    """Single-panel gap plot from one history.json."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    ax.set_facecolor(_BG)
    _panel(ax, _load(history_path), title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=_BG)
    return out_path


def plot_trajectory(eval_logs: dict[str, str], out_path: str) -> str:
    """Sealed landing rate per evaluation, one line per agent, from each agent's eval_log.jsonl."""
    groups: dict[str, list[float]] = {}
    for name, path in eval_logs.items():
        with open(path) as fh:
            groups[name] = [json.loads(line)["landing_rate"] for line in fh if line.strip()]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    ax.set_facecolor(_BG)
    for index, (name, rates) in enumerate(groups.items()):
        color, marker = _AGENTS[index % len(_AGENTS)]
        ax.plot(range(1, len(rates) + 1), rates, color=color, marker=marker, markersize=8, linewidth=3, label=name)
    ax.axhline(0.9, color="#888888", linestyle=":", linewidth=2, label="target 0.90")
    ax.set_title("Sealed landing rate per evaluation", fontsize=15, fontweight="bold", color="w", pad=12)
    ax.set_xlabel("evaluation", fontsize=12, color="#aaaaaa")
    ax.set_ylabel("landing rate (sealed)", fontsize=12, color=_LAND, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors="#aaaaaa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="w", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=_BG)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the verification-gap figure.")
    parser.add_argument("--naive", required=True, help="naive run history.json")
    parser.add_argument("--fixed", required=True, help="fixed run history.json")
    parser.add_argument("--out", default="artifacts/verification_gap.png")
    args = parser.parse_args()
    print("wrote", plot(args.naive, args.fixed, args.out))


if __name__ == "__main__":
    main()
