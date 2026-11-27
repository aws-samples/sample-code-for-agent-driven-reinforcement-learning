"""A trained policy's actor network as plain numpy arrays, so the grader needs no torch.

Loading SB3's `policy.zip` would unpickle agent-produced data; these arrays load with `allow_pickle=False`.
"""

from __future__ import annotations

import io
import os
import pathlib
import zipfile

import numpy as np

WEIGHTS_FILE = "policy_weights.npz"

# SB3 MlpPolicy defaults: 8 inputs, two tanh layers of 64, 4 actions. Weights are (out, in), as in torch.
SHAPES = {
    "w0": (64, 8),
    "b0": (64,),
    "w1": (64, 64),
    "b1": (64,),
    "w2": (4, 64),
    "b2": (4,),
}

_MAX_UNCOMPRESSED_BYTES = 1_000_000

Weights = dict[str, np.ndarray]


def export_weights(model, path: str | os.PathLike) -> None:
    """Save an SB3 PPO model's actor as `.npz`."""
    policy = model.policy
    linears = [layer for layer in policy.mlp_extractor.policy_net if hasattr(layer, "weight")]
    linears.append(policy.action_net)
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    _validate(arrays)
    np.savez(path, **arrays)


def load_weights(source: str | os.PathLike | bytes) -> Weights:
    """Load and validate weights from a path or `.npz` bytes. Raises ValueError if invalid."""
    data = source if isinstance(source, bytes) else pathlib.Path(source).read_bytes()
    buffer = io.BytesIO(data)
    try:
        with zipfile.ZipFile(buffer) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ValueError("weights are not a valid .npz file") from exc

    names = {info.filename.removesuffix(".npy") for info in infos}
    if names != set(SHAPES) or len(infos) != len(SHAPES):
        raise ValueError(f"weights must contain exactly {sorted(SHAPES)}, got {sorted(names)}")
    if sum(info.file_size for info in infos) > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError("weights file is too large")

    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in SHAPES}
    _validate(arrays)
    return {name: array.astype(np.float32) for name, array in arrays.items()}


def _validate(arrays: Weights) -> None:
    for name, shape in SHAPES.items():
        array = arrays[name]
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} has dtype {array.dtype}, expected a float type")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values")


def logits(weights: Weights, obs: np.ndarray) -> np.ndarray:
    """Logits for one observation (8,) or a batch (n, 8)."""
    h = np.asarray(obs, dtype=np.float32)
    h = np.tanh(h @ weights["w0"].T + weights["b0"])
    h = np.tanh(h @ weights["w1"].T + weights["b1"])
    return h @ weights["w2"].T + weights["b2"]


def act(weights: Weights, obs: np.ndarray) -> int:
    """Deterministic action (argmax), matching SB3 `predict(deterministic=True)`."""
    return int(np.argmax(logits(weights, obs)))
