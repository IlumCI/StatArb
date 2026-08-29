"""Tensor assembly for the neural overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.features.dataset import FEATURE_COLUMNS

#: Features the gating network sees. Deliberately small and interpretable: these are
#: the variables that define "what regime are we in", not the return forecast itself.
GATE_COLUMNS: tuple[str, ...] = (
    "leader_vol",
    "own_vol",
    "staleness",
    "illiq_z",
    "hawkes_ratio",
)


def build_matrices(
    frame: pd.DataFrame,
    residual: pd.Series,
    hawkes_ratio: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Return (features, gate features, target, index) with NaN rows dropped.

    ``residual`` is the baseline's prediction error, which is what the net learns.
    """
    work = frame[list(FEATURE_COLUMNS)].copy()
    work["hawkes_ratio"] = (
        hawkes_ratio.reindex(frame.index) if hawkes_ratio is not None else 1.0
    )
    work["__target__"] = residual.reindex(frame.index)

    usable = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["__target__"])
    if usable.empty:
        empty = np.empty((0, len(FEATURE_COLUMNS)))
        return empty, np.empty((0, len(GATE_COLUMNS))), np.empty(0), usable.index

    x = usable[list(FEATURE_COLUMNS)].to_numpy(dtype="float32")
    g = usable[list(GATE_COLUMNS)].to_numpy(dtype="float32")
    y = usable["__target__"].to_numpy(dtype="float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return x, g, y, usable.index
