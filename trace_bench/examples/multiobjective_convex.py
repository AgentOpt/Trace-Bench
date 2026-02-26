"""Multi-objective SixHumpCamel convex function task (base_loss + reg_loss).

Optimizes a 2D input x = [x1, x2] to minimize both the base Six-Hump Camel
function value and an L2-squared regularization term independently.
"""
from __future__ import annotations

import copy
import re
from typing import Dict, Tuple

import numpy as np

from opto import trace
from opto.trainer.guide import Guide
from opto.trainer.objectives import ObjectiveConfig


# ---------------------------------------------------------------------------
# Random seed helper (from OpenTrace convex examples)
# ---------------------------------------------------------------------------

def _np_random(seed: int | None = None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError(f"Seed must be a non-negative integer, got {seed!r}")
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


# ---------------------------------------------------------------------------
# Norm helper
# ---------------------------------------------------------------------------

def _norm_term(x: np.ndarray, norm_coef: float) -> float:
    if norm_coef == 0.0:
        return 0.0
    return float(norm_coef * (x[0] ** 2 + x[1] ** 2))


# ---------------------------------------------------------------------------
# SixHumpCamel environment (self-contained, no cvxpy dependency)
# ---------------------------------------------------------------------------

def _six_hump_camel_base(x: np.ndarray) -> float:
    u, v = float(x[0]), float(x[1])
    return float((4 - 2.1 * u**2 + (u**4) / 3) * u**2 + u * v + (-4 + 4 * v**2) * v**2)


# Known global minimum of unregularized Six-Hump Camel is approx -1.0316
# With norm_coef=1.0 l2sq, the minimum shifts; we use a pre-computed value.
_SHC_CERT_NORM1 = -0.2155  # SOS certificate for norm_coef=1.0 (order_d=3, SCS)


class SixHumpCamelEnv:
    """Minimal Six-Hump Camel environment for benchmarking.

    The agent picks x = [x1, x2] in [-2, 2]^2 to minimize:
        total_loss = base_loss(x) + norm_coef * ||x||_2^2

    The environment reports base_loss and reg_loss separately for
    multi-objective scoring.
    """

    def __init__(self, horizon: int = 200, norm_coef: float = 1.0, seed: int = 42):
        self.x_low = -2.0
        self.x_high = 2.0
        self.horizon = horizon
        self.norm_coef = float(norm_coef)
        self.left_attempts = horizon
        self.prev_x = None
        self.called_reset = False
        self.stop_keywords = ["reach", "stay", "stop"]

        self._np_random, _ = _np_random(seed)

        # Pre-computed certificate (SOS lower bound)
        if norm_coef == 1.0:
            self.certificate_y = _SHC_CERT_NORM1
        elif norm_coef == 0.0:
            self.certificate_y = -1.0316
        else:
            # Conservative fallback: evaluate on a grid
            self.certificate_y = self._grid_min(steps=200)

        self.docstring = (
            "You are trying to minimize an objective by choosing x = [x1, x2].\n"
            f"total_loss = f(x) + {self.norm_coef} * (x1^2 + x2^2)\n"
            f"x1 and x2 must be in [{self.x_low}, {self.x_high}].\n"
            f"You have {self.horizon} attempts."
        )

    def _grid_min(self, steps: int = 200) -> float:
        best = float("inf")
        for i in range(steps + 1):
            for j in range(steps + 1):
                x1 = self.x_low + (self.x_high - self.x_low) * i / steps
                x2 = self.x_low + (self.x_high - self.x_low) * j / steps
                x = np.array([x1, x2])
                val = _six_hump_camel_base(x) + _norm_term(x, self.norm_coef)
                if val < best:
                    best = val
        return best

    def reset(self, **kwargs):
        x = self._np_random.uniform(self.x_low, self.x_high, size=2)
        x = np.round(x, 4)
        self.prev_x = x
        self.left_attempts = self.horizon

        base = _six_hump_camel_base(x)
        reg = _norm_term(x, self.norm_coef)
        total = base + reg

        obs = (
            f"Function outputs total y = {total}\n"
            f"  base f(x) = {base}\n"
            f"  regularizer = {reg}  (coef={self.norm_coef})\n"
            f"You have {self.left_attempts} attempts left!\n"
            "Please output the next x that will minimize y.\n"
            "Format: x = [x1, x2]\nOutput:"
        )
        self.called_reset = True
        return obs

    def _text_extract(self, text: str):
        for kw in self.stop_keywords:
            if kw in text:
                return None, True
        pattern = r"\[(-?\d+\.?\d*(?:e[-+]?\d+)?),\s*(-?\d+\.?\d*(?:e[-+]?\d+)?)\]"
        match = re.search(pattern, text)
        if match is None:
            return None, False
        return np.array([float(g) for g in match.groups()], dtype=float), False

    def step(self, action: str):
        if not self.called_reset:
            raise RuntimeError("Must call env.reset() before step()")

        x, stop = self._text_extract(action)

        if x is None and not stop:
            return None, -1.0, True, {
                "feedback": f"Invalid action: {action}",
                "base_loss": None, "reg_loss": None, "total_loss": None,
            }

        if stop:
            base = _six_hump_camel_base(self.prev_x)
            reg = _norm_term(self.prev_x, self.norm_coef)
            total = base + reg
            return None, float(-total), True, {
                "feedback": f"Stopped at {self.prev_x.tolist()}.",
                "base_loss": base, "reg_loss": reg, "total_loss": total,
            }

        if np.any(x < self.x_low) or np.any(x > self.x_high):
            base = _six_hump_camel_base(self.prev_x)
            reg = _norm_term(self.prev_x, self.norm_coef)
            total = base + reg
            return None, float(-total), True, {
                "feedback": f"x out of range [{self.x_low}, {self.x_high}].",
                "base_loss": base, "reg_loss": reg, "total_loss": total,
            }

        base = _six_hump_camel_base(x)
        reg = _norm_term(x, self.norm_coef)
        total = base + reg
        self.prev_x = x
        self.left_attempts -= 1

        done = self.left_attempts <= 0
        obs = (
            f"Function outputs total y = {total}\n"
            f"  base f(x) = {base}\n"
            f"  regularizer = {reg}  (coef={self.norm_coef})\n"
            f"You have {self.left_attempts} attempts left!\n"
            "Format: x = [x1, x2]\nOutput:"
        )
        return obs, float(-total), done, {
            "feedback": "Choose x to minimize y.",
            "base_loss": base, "reg_loss": reg, "total_loss": total,
        }


# ---------------------------------------------------------------------------
# Multi-objective Guide
# ---------------------------------------------------------------------------

class ConvexRewardGuide(Guide):
    """Scores by base_loss and reg_loss independently (both minimize)."""

    def __init__(self, env: SixHumpCamelEnv):
        self.env = env

    def _score_on_copy(self, action: str):
        env_copy = copy.deepcopy(self.env)
        return env_copy.step(action)

    def get_feedback(self, query, response, reference=None, **kwargs) -> Tuple[float, str]:
        obs, reward, done, info = self.env.step(str(response))
        feedback = ((obs + "\n\n") if obs else "") + info.get("feedback", "")
        return float(reward), feedback

    def get_score_dict(self, query, response, reference=None, **kwargs) -> Dict[str, float]:
        obs, reward, done, info = self._score_on_copy(str(response))
        base_loss = info.get("base_loss")
        reg_loss = info.get("reg_loss")
        if base_loss is None or reg_loss is None:
            base_loss = float("inf")
            reg_loss = float("inf")
        return {"base_loss": float(base_loss), "reg_loss": float(reg_loss)}


# ---------------------------------------------------------------------------
# Trace-Bench entry point
# ---------------------------------------------------------------------------

def build_trace_problem(**_override_eval_kwargs):
    env = SixHumpCamelEnv(horizon=200, norm_coef=1.0, seed=42)
    instruction = env.reset()
    initial_input = instruction.split("\n")[0].strip()

    param = trace.node(
        initial_input,
        description="Input x into the hidden function to get y.",
        trainable=True,
    )

    guide = ConvexRewardGuide(env)

    objective_config = ObjectiveConfig(
        mode="weighted",
        weights={"base_loss": 1.0, "reg_loss": 1.0},
        minimize=frozenset({"base_loss", "reg_loss"}),
        seed=42,
    )

    return dict(
        param=param,
        guide=guide,
        train_dataset=dict(inputs=[None], infos=[None]),
        optimizer_kwargs=dict(
            objective="Minimize the function by choosing x = [x1, x2].",
            memory_size=10,
        ),
        metadata=dict(benchmark="multiobjective", entry="convex_sixhumpcamel"),
        objective_config=objective_config,
    )


__all__ = ["build_trace_problem", "SixHumpCamelEnv", "ConvexRewardGuide"]
