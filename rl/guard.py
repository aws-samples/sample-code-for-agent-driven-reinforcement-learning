"""AST allowlist and restricted-builtins loader for agent-authored reward functions."""

from __future__ import annotations

import ast
import builtins
import math

import numpy as np

REWARD_FN_NAME = "reward"
REQUIRED_PARAMS = ("obs", "action", "next_obs", "terminated", "truncated", "info")

_ALLOWED_BUILTINS = ("abs", "min", "max", "sum", "len", "float", "int", "pow", "round", "bool")
_ALLOWED_MODULES = ("math", "np")
# Allowlisted rather than denylisted: numpy exposes file I/O, pickle loading and ctypes, and grows
# with every release.
_ALLOWED_CALLS = frozenset(
    """
    math.sqrt math.exp math.log math.tanh math.sin math.cos math.atan2 math.hypot math.fabs math.isfinite
    np.abs np.sqrt np.clip np.tanh np.exp np.log np.square np.sign np.minimum np.maximum np.hypot
    np.isfinite np.sum np.mean np.array np.asarray np.linalg.norm
    """.split()
)
_ALLOWED_CONSTANTS = frozenset({"math.pi", "math.e", "math.inf", "np.pi", "np.inf"})


def _dotted(node: ast.expr) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class RewardGuardError(ValueError):
    """Raised when generated reward source violates the allowlist or signature contract."""


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append("imports are not allowed in reward functions")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append("imports are not allowed in reward functions")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.errors.append(f"access to dunder/private attribute '{node.attr}' is not allowed")
        dotted = _dotted(node)
        if dotted and dotted.split(".")[0] in _ALLOWED_MODULES:
            if dotted not in _ALLOWED_CONSTANTS:
                self.errors.append(f"'{dotted}' is not allowed")
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.errors.append(f"access to name '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in _ALLOWED_BUILTINS:
                self.errors.append(f"call to '{func.id}' is not allowed")
        elif isinstance(func, ast.Attribute):
            dotted = _dotted(func)
            if dotted is None:
                self.errors.append("calling a method on an expression result is not allowed")
                self.visit(func.value)
            elif dotted.split(".")[0] not in _ALLOWED_MODULES:
                self.errors.append(f"calling methods on '{dotted.split('.')[0]}' is not allowed")
            elif dotted not in _ALLOWED_CALLS:
                self.errors.append(f"call to '{dotted}' is not allowed")
        else:
            self.errors.append("only named functions can be called")
            self.visit(func)
        for child in [*node.args, *node.keywords]:
            self.visit(child)


def _check_signature(tree: ast.Module) -> list[str]:
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == REWARD_FN_NAME]
    if not defs:
        return [f"source must define a function named '{REWARD_FN_NAME}'"]
    params = tuple(a.arg for a in defs[-1].args.args)
    if params != REQUIRED_PARAMS:
        return [f"'{REWARD_FN_NAME}' must take exactly {REQUIRED_PARAMS}, got {params}"]
    return []


def validate(source: str) -> None:
    """Raise RewardGuardError if the source is unsafe or has the wrong signature."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RewardGuardError(f"syntax error: {exc}") from exc

    errors = _check_signature(tree)
    validator = _Validator()
    validator.visit(tree)
    errors.extend(validator.errors)
    if errors:
        raise RewardGuardError("; ".join(errors))


def load_reward(source: str):
    """Validate `source` and return the `reward` callable in a restricted namespace."""
    validate(source)
    safe_builtins = {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}
    namespace: dict = {"__builtins__": safe_builtins, "math": math, "np": np}
    exec(compile(source, "<generated_reward>", "exec"), namespace)  # noqa: S102 - sandboxed namespace
    return namespace[REWARD_FN_NAME]


def load_reward_file(path: str):
    with open(path, encoding="utf-8") as fh:
        return load_reward(fh.read())
