"""Linear lead-lag baseline.

This is the benchmark every other component must beat. It is deliberately simple:
a ridge regression of each follower's forward return on causal leader and
microstructure features, fitted on a trailing window and applied out of sample.

Keeping it simple is the point. A neural net that cannot beat this is not adding
information, and the blending logic in ``models/nn/blend.py`` relies on having an
honest, stable reference to measure against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.features.dataset import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class StandardScaler:
    """Mean/scale fitted on training data only."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        self.mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        # A constant column carries no information; scaling by 1 leaves it at zero.
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("scaler not fitted")
        return np.nan_to_num((x - self.mean) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalised intercept.

    Solved via ``lstsq`` on the augmented normal equations rather than an explicit
    inverse, which stays stable when features are near-collinear (leader_cum_2 and
    leader_cum_6 overlap heavily by construction).
    """
    n, k = x.shape
    xc = np.hstack([np.ones((n, 1)), x])
    penalty = np.eye(k + 1) * alpha
    penalty[0, 0] = 0.0
    gram = xc.T @ xc + penalty
    rhs = xc.T @ y
    coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    return coef


@dataclass
class LeadLagBaseline:
    """Per-asset ridge on shared features, with a pooled fallback.

    Assets with too little history fall back to coefficients pooled across the whole
    cross-section rather than being dropped, so a newly listed follower still trades
    on the basket's average relationship instead of on noise.
    """

    alpha: float = 10.0
    min_train_rows: int = 500
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    target_column: str = TARGET_COLUMN
    coefs: dict[str, np.ndarray] = field(default_factory=dict)
    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    pooled_coef: np.ndarray | None = None
    pooled_scaler: StandardScaler | None = None
    fitted_symbols: tuple[str, ...] = ()

    def _matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[list(self.feature_columns)].to_numpy(dtype="float64")
        y = frame[self.target_column].to_numpy(dtype="float64") if self.target_column in frame else np.array([])
        return x, y

    def fit(self, frame: pd.DataFrame) -> LeadLagBaseline:
        """Fit on a long-format training frame indexed by (timestamp, symbol)."""
        usable = frame.dropna(subset=[self.target_column])
        if usable.empty:
            raise ValueError("no rows with a target to fit on")

        x_all, y_all = self._matrix(usable)
        self.pooled_scaler = StandardScaler().fit(x_all)
        self.pooled_coef = ridge_fit(self.pooled_scaler.transform(x_all), y_all, self.alpha)

        self.coefs, self.scalers = {}, {}
        for sym, grp in usable.groupby(level="symbol", observed=True):
            if len(grp) < self.min_train_rows:
                continue
            x, y = self._matrix(grp)
            scaler = StandardScaler().fit(x)
            self.coefs[sym] = ridge_fit(scaler.transform(x), y, self.alpha)
            self.scalers[sym] = scaler
        self.fitted_symbols = tuple(sorted(self.coefs))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predicted forward return for every row, indexed like ``frame``."""
        if self.pooled_coef is None or self.pooled_scaler is None:
            raise RuntimeError("baseline not fitted")
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        for sym, grp in frame.groupby(level="symbol", observed=True):
            x, _ = self._matrix(grp)
            if sym in self.coefs:
                coef, scaler = self.coefs[sym], self.scalers[sym]
            else:
                coef, scaler = self.pooled_coef, self.pooled_scaler
            xs = scaler.transform(x)
            out.loc[grp.index] = coef[0] + xs @ coef[1:]
        return out

    def coefficient_table(self) -> pd.DataFrame:
        """Per-asset coefficients, for inspecting what the model actually learned."""
        rows = {}
        for sym, coef in self.coefs.items():
            rows[sym] = dict(zip(("intercept", *self.feature_columns), coef, strict=True))
        if self.pooled_coef is not None:
            rows["__pooled__"] = dict(
                zip(("intercept", *self.feature_columns), self.pooled_coef, strict=True)
            )
        return pd.DataFrame(rows).T
