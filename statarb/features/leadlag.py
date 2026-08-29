"""Lead-lag measurement and feature construction.

The core hypothesis: the leader impounds common information first and illiquid
followers reprice with a delay. This module both *measures* that (research) and
*builds features* from it (production), keeping the two clearly separated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeadLagStat:
    symbol: str
    n: int
    lag0: float
    lags: dict[int, float]
    t_lag1: float
    reverse_lag1: float

    @property
    def best_lag(self) -> int:
        return max(self.lags, key=lambda k: abs(self.lags[k])) if self.lags else 0


def _corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 100:
        return float("nan"), len(both)
    c = both.iloc[:, 0].corr(both.iloc[:, 1])
    return float(c), len(both)


def cross_correlation(
    leader_returns: pd.Series, follower_returns: pd.DataFrame, max_lag: int = 4
) -> pd.DataFrame:
    """Correlation of each follower's return at ``t`` with the leader's at ``t-k``.

    Also reports the reverse direction (follower leading leader) at lag 1. A genuine
    lead-lag shows a positive forward coefficient and a near-zero reverse one; if both
    are large the two series are simply co-moving and there is nothing to trade.
    """
    rows = []
    for sym in follower_returns.columns:
        f = follower_returns[sym]
        lags: dict[int, float] = {}
        n1 = 0
        for k in range(max_lag + 1):
            c, n = _corr(leader_returns.shift(k), f)
            lags[k] = c
            if k == 1:
                n1 = n
        c1 = lags.get(1, float("nan"))
        t1 = c1 * np.sqrt(max(n1 - 2, 1)) / np.sqrt(max(1e-12, 1 - c1**2)) if np.isfinite(c1) else float("nan")
        rev, _ = _corr(f.shift(1), leader_returns)
        rows.append(
            {
                "symbol": sym,
                "n": n1,
                **{f"lag{k}": v for k, v in lags.items()},
                "t_lag1": t1,
                "reverse_lag1": rev,
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def hayashi_yoshida(x: pd.Series, y: pd.Series, lag: int = 0) -> float:
    """Hayashi-Yoshida style lagged covariance, normalised to a correlation.

    Designed for non-synchronous observations, which is the regime these illiquid
    followers actually trade in. Here it acts as a robustness check on the plain
    Pearson lag structure rather than as a separate signal.
    """
    a = x.shift(lag)
    both = pd.concat([a, y], axis=1).dropna()
    if len(both) < 50:
        return float("nan")
    u, v = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    denom = np.sqrt((u**2).sum() * (v**2).sum())
    return float((u * v).sum() / denom) if denom > 0 else float("nan")


def leader_features(
    leader_open: pd.Series,
    leader_close: pd.Series,
    lags: tuple[int, ...],
    vol_window: int,
) -> pd.DataFrame:
    """Causal leader features known at the close of each bar.

    Every column is a function of bars at or before ``t``. The intrabar (open-to-close)
    return is used rather than close-to-close because it is the move a trader could
    actually have observed completing within the bar.
    """
    o = leader_open.where(leader_open > 0)
    c = leader_close.where(leader_close > 0)
    intrabar = np.log(c / o)
    c2c = np.log(c).diff()

    feats = {"leader_intrabar": intrabar, "leader_c2c": c2c}
    for k in lags:
        # Cumulative leader move over the k bars ending at t.
        feats[f"leader_cum_{k}"] = c2c.rolling(k, min_periods=1).sum()
    vol = c2c.rolling(vol_window, min_periods=max(2, vol_window // 4)).std()
    feats["leader_vol"] = vol
    feats["leader_intrabar_z"] = intrabar / vol.replace(0.0, np.nan)
    return pd.DataFrame(feats)


def trailing_quantile_gate(
    series: pd.Series, window: int, quantile: float, min_periods: int
) -> pd.Series:
    """Trailing quantile of |series|, shifted so bar ``t`` uses only data before ``t``.

    This replaces the full-sample quantile used in exploratory work. Using a
    full-sample threshold leaks the future distribution of leader moves into every
    historical decision and inflates measured performance.
    """
    absolute = series.abs()
    q = absolute.rolling(window, min_periods=min_periods).quantile(quantile)
    return q.shift(1)


def gate_mask(series: pd.Series, window: int, quantile: float, min_periods: int) -> pd.Series:
    """True where |series| exceeds its trailing quantile: a tradable leader move."""
    threshold = trailing_quantile_gate(series, window, quantile, min_periods)
    return (series.abs() >= threshold) & threshold.notna()
