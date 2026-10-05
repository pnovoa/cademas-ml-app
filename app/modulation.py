"""Contextual modulation operators for CADEMAS-ML priority scores.

Operators map the integrated predictive score (d) and the contextual
alignment score (c) to a priority score (p):
  - Linear:    p = λ·d + (1−λ)·c
  - Geometric: p = d^λ · c^(1−λ)
  - Minimum:   p = min{d, c}   (parameter-free)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

OPERATOR_LINEAR = "linear"
OPERATOR_GEOMETRIC = "geometric"
OPERATOR_MINIMUM = "minimum"

OPERATOR_KEYS = (OPERATOR_LINEAR, OPERATOR_GEOMETRIC, OPERATOR_MINIMUM)

OPERATOR_LABELS = {
    OPERATOR_LINEAR: "Linear",
    OPERATOR_GEOMETRIC: "Geometric",
    OPERATOR_MINIMUM: "Minimum",
}

OPERATOR_ORDER = {
    OPERATOR_LINEAR: 0,
    OPERATOR_GEOMETRIC: 1,
    OPERATOR_MINIMUM: 2,
}

USES_LAMBDA = frozenset({OPERATOR_LINEAR, OPERATOR_GEOMETRIC})


def normalize_operator(operator: str | None) -> str:
    """Map UI labels or keys to a canonical operator key."""
    if operator is None:
        return OPERATOR_LINEAR
    key = str(operator).strip().lower()
    aliases = {
        "linear": OPERATOR_LINEAR,
        "geometric": OPERATOR_GEOMETRIC,
        "geom": OPERATOR_GEOMETRIC,
        "minimum": OPERATOR_MINIMUM,
        "min": OPERATOR_MINIMUM,
    }
    if key not in aliases:
        raise ValueError(f"Unknown modulation operator: {operator!r}")
    return aliases[key]


def operator_display_label(operator: str) -> str:
    return OPERATOR_LABELS[normalize_operator(operator)]


def uses_lambda(operator: str) -> bool:
    return normalize_operator(operator) in USES_LAMBDA


def operator_formula_caption(operator: str) -> str:
    op = normalize_operator(operator)
    if op == OPERATOR_LINEAR:
        return (
            "Linear: p = λ·d + (1−λ)·c. "
            "Weighted average of the integrated predictive score and contextual "
            "alignment; high λ favors the predictive signal."
        )
    if op == OPERATOR_GEOMETRIC:
        return (
            "Geometric: p = d^λ · c^(1−λ). "
            "Weighted geometric mean; a near-zero input pulls the priority score toward zero."
        )
    return (
        "Minimum: p = min{d, c}. "
        "Bottleneck / limiting-factor operator; parameter-free (λ not used)."
    )


def modulate(
    ri: float | np.ndarray | pd.Series,
    ci: float | np.ndarray | pd.Series,
    operator: str,
    lambda_val: float = 0.5,
) -> float | np.ndarray | pd.Series:
    """Apply the selected contextual modulation operator."""
    op = normalize_operator(operator)
    lam = float(np.clip(lambda_val, 0.0, 1.0))

    if op == OPERATOR_LINEAR:
        return lam * ri + (1.0 - lam) * ci

    if op == OPERATOR_GEOMETRIC:
        # Safe handling of zeros: a^0 = 1 for a >= 0; 0^positive = 0.
        ri_arr = np.asarray(ri, dtype=float)
        ci_arr = np.asarray(ci, dtype=float)
        ri_safe = np.clip(ri_arr, 0.0, None)
        ci_safe = np.clip(ci_arr, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.power(ri_safe, lam) * np.power(ci_safe, 1.0 - lam)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(ri, pd.Series):
            return pd.Series(score, index=ri.index, name=ri.name)
        if np.isscalar(ri) and np.isscalar(ci):
            return float(score)
        return score

    # Minimum
    if isinstance(ri, pd.Series) or isinstance(ci, pd.Series):
        return pd.concat([pd.Series(ri), pd.Series(ci)], axis=1).min(axis=1)
    return np.minimum(ri, ci)


def additive_contributions(
    ri: float | np.ndarray | pd.Series,
    ci: float | np.ndarray | pd.Series,
    operator: str,
    lambda_val: float = 0.5,
) -> tuple[float | np.ndarray | pd.Series | None, float | np.ndarray | pd.Series | None]:
    """Return additive ML/Context contributions only for the Linear operator."""
    if normalize_operator(operator) != OPERATOR_LINEAR:
        return None, None
    lam = float(np.clip(lambda_val, 0.0, 1.0))
    return lam * ri, (1.0 - lam) * ci


@dataclass(frozen=True)
class ModulationConfig:
    operator: str
    lambda_val: float | None
    label: str

    @property
    def operator_label(self) -> str:
        return operator_display_label(self.operator)


def config_label(operator: str, lambda_val: float | None = None) -> str:
    op = normalize_operator(operator)
    if op == OPERATOR_MINIMUM or lambda_val is None:
        return OPERATOR_LABELS[OPERATOR_MINIMUM]
    return f"{OPERATOR_LABELS[op]} (λ={float(lambda_val):.2f})"


def build_q_mod(
    operators: Iterable[str],
    lambda_steps: Sequence[float] | None = None,
) -> list[ModulationConfig]:
    """Build the ordered set of modulation configurations Q_mod.

    Order: Linear by ascending λ, then Geometric by ascending λ, then Minimum.
    """
    selected = []
    seen = set()
    for raw in operators:
        key = normalize_operator(raw)
        if key not in seen:
            selected.append(key)
            seen.add(key)

    steps = list(lambda_steps) if lambda_steps is not None else [0.5]
    configs: list[ModulationConfig] = []

    for op in (OPERATOR_LINEAR, OPERATOR_GEOMETRIC):
        if op not in seen:
            continue
        for lam in steps:
            configs.append(
                ModulationConfig(
                    operator=op,
                    lambda_val=float(lam),
                    label=config_label(op, lam),
                )
            )

    if OPERATOR_MINIMUM in seen:
        configs.append(
            ModulationConfig(
                operator=OPERATOR_MINIMUM,
                lambda_val=None,
                label=config_label(OPERATOR_MINIMUM),
            )
        )

    return configs


def default_top_n_config(
    q_mod: Sequence[ModulationConfig],
    global_operator: str,
    preferred_lambda: float = 0.5,
) -> ModulationConfig | None:
    """Pick the default Top-N configuration from Q_mod.

    Preference:
      1. Global Linear/Geometric (if present) at λ closest to preferred_lambda
      2. Minimum if present
      3. First available configuration
    """
    if not q_mod:
        return None

    global_op = normalize_operator(global_operator)
    if global_op in USES_LAMBDA:
        matching = [c for c in q_mod if c.operator == global_op and c.lambda_val is not None]
        if matching:
            return min(matching, key=lambda c: abs(float(c.lambda_val) - preferred_lambda))

    for c in q_mod:
        if c.operator == OPERATOR_MINIMUM:
            return c

    return q_mod[0]


def find_config_by_label(
    q_mod: Sequence[ModulationConfig],
    label: str,
) -> ModulationConfig | None:
    for c in q_mod:
        if c.label == label:
            return c
    return None
