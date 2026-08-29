"""Blending the neural overlay with the linear baseline.

The blend weight is *earned*, not assumed. It is driven by the net's realised
out-of-sample information coefficient: a net whose predictions do not correlate with
subsequent residuals contributes nothing and the signal collapses to the baseline.

This is the guardrail that makes it safe to add a flexible model to a thin-edge
strategy. Without it, a net that has merely memorised noise would silently take over
the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def information_coefficient(pred: pd.Series, actual: pd.Series) -> float:
    """Spearman rank correlation between prediction and outcome."""
    both = pd.concat([pred.rename("p"), actual.rename("a")], axis=1).dropna()
    if len(both) < 30 or both["p"].nunique() < 5:
        return float("nan")
    return float(both["p"].corr(both["a"], method="spearman"))


@dataclass
class BlendState:
    """Rolling record of how much the net has actually been earning."""

    min_baseline_weight: float = 0.5
    ic_window: int = 2000
    _preds: list = field(default_factory=list, repr=False)
    _actuals: list = field(default_factory=list, repr=False)
    last_ic: float = float("nan")

    def update(self, pred: pd.Series, actual: pd.Series) -> float:
        self._preds.append(pred)
        self._actuals.append(actual)
        p = pd.concat(self._preds[-50:])
        a = pd.concat(self._actuals[-50:])
        if len(p) > self.ic_window:
            p, a = p.iloc[-self.ic_window :], a.iloc[-self.ic_window :]
        self.last_ic = information_coefficient(p, a)
        return self.last_ic

    @property
    def nn_weight(self) -> float:
        """Weight on the neural residual, in [0, 1 - min_baseline_weight].

        A negative or missing IC means the net is worse than useless, so it gets zero
        weight rather than being trusted to be contrarian.
        """
        ic = self.last_ic
        if not np.isfinite(ic) or ic <= 0:
            return 0.0
        headroom = 1.0 - self.min_baseline_weight
        # Scale so an IC of 0.05 (strong for hourly cross-sectional data) earns roughly
        # full headroom; anything smaller is discounted proportionally.
        return float(np.clip(ic / 0.05, 0.0, 1.0) * headroom)


def blend(baseline: pd.Series, nn_residual: pd.Series, nn_weight: float) -> pd.Series:
    """Combine baseline prediction with the net's residual correction."""
    if nn_weight <= 0:
        return baseline
    correction = nn_residual.reindex(baseline.index).fillna(0.0)
    return baseline + nn_weight * correction
