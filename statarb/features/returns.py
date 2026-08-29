"""Return construction.

The distinction between close-to-close and open-to-close returns is the whole ball
game for lead-lag. Close-to-close overstates the tradable edge because the signal is
formed at the same close it is measured against; the strategy can only ever capture
the open-to-close leg of the *next* bar. Both are provided, and the backtest is wired
to the tradable one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Close-to-close log returns. Research use; not directly tradable."""
    safe = close.where(close > 0)
    return np.log(safe).diff()


def intrabar_returns(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Open-to-close log return within each bar.

    This is the tradable unit: enter at a bar's open, exit at its close.
    """
    o = open_.where(open_ > 0)
    c = close.where(close > 0)
    return np.log(c / o)


def forward_return(
    open_: pd.DataFrame, close: pd.DataFrame, *, entry_lag: int = 1, holding_bars: int = 1
) -> pd.DataFrame:
    """Return earned by a decision made at the close of bar ``t``.

    Enters at the open of bar ``t + entry_lag`` and exits at the close of bar
    ``t + entry_lag + holding_bars - 1``. With the defaults this is "decide on this
    close, buy the next open, sell that bar's close".

    The result is indexed by the *decision* bar ``t``, so it is a strictly forward
    looking target that must never be used as a feature.
    """
    if entry_lag < 1:
        raise ValueError("entry_lag must be >= 1; entering at the signal bar's own close is lookahead")
    if holding_bars < 1:
        raise ValueError("holding_bars must be >= 1")
    entry = open_.shift(-entry_lag)
    exit_ = close.shift(-(entry_lag + holding_bars - 1))
    return np.log(exit_.where(exit_ > 0) / entry.where(entry > 0))


def realised_vol(returns: pd.DataFrame | pd.Series, window: int, min_periods: int | None = None):
    """Trailing realised volatility (per bar)."""
    mp = min_periods if min_periods is not None else max(2, window // 4)
    return returns.rolling(window, min_periods=mp).std()


def winsorise(frame: pd.DataFrame, window: int, z: float, min_periods: int | None = None) -> pd.DataFrame:
    """Clip returns at +/- z trailing standard deviations.

    Trailing rather than full-sample, so a future crash cannot influence how an
    earlier bar is clipped. Guards against Yahoo's occasional bad prints without
    quietly deleting the genuine jumps the Hawkes model needs to see.
    """
    mp = min_periods if min_periods is not None else max(2, window // 4)
    sigma = frame.rolling(window, min_periods=mp).std()
    limit = (z * sigma).shift(1)
    return frame.clip(lower=-limit, upper=limit, axis=None).where(limit.notna(), frame)


def zscore(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing z-score, shifted so bar ``t`` uses only data up to ``t-1``."""
    mp = min_periods if min_periods is not None else max(2, window // 4)
    mu = frame.rolling(window, min_periods=mp).mean().shift(1)
    sd = frame.rolling(window, min_periods=mp).std().shift(1)
    return (frame - mu) / sd.replace(0.0, np.nan)


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank each bar's cross-section into [-1, 1], NaN-safe.

    Uses only contemporaneous values across assets, so it introduces no time leakage.
    """
    ranked = frame.rank(axis=1, pct=True, na_option="keep")
    return (ranked - 0.5) * 2.0
