"""Microstructure features.

These describe *how hard an asset is to trade*, which for this strategy matters as
much as the return forecast: the lead-lag signal is strongest exactly where execution
is worst, and these features are what let the system see that trade-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    return close.mul(volume)


def amihud_illiquidity(
    returns: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    """Amihud (2002) illiquidity: mean |return| per dollar of volume.

    Computed as a ratio of rolling sums rather than a rolling mean of per-bar ratios.
    Yahoo zeroes the volume on about half of hourly crypto bars, which would make the
    per-bar ratio infinite; aggregating first keeps the estimate finite and stable.
    """
    mp = min_periods if min_periods is not None else max(2, window // 4)
    dv = dollar_volume(close, volume)
    num = returns.abs().rolling(window, min_periods=mp).sum()
    den = dv.rolling(window, min_periods=mp).sum()
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def rolling_dollar_volume(
    close: pd.DataFrame, volume: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    """Trailing mean dollar volume per bar. The liquidity input to the cost model."""
    mp = min_periods if min_periods is not None else max(2, window // 4)
    return dollar_volume(close, volume).rolling(window, min_periods=mp).mean()


def staleness_ratio(close: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing fraction of bars whose close did not move.

    High staleness means the printed price is an old quote, so measured lead-lag on
    that name is partly an artefact rather than a fill you could actually get.
    """
    mp = min_periods if min_periods is not None else max(2, window // 4)
    flat = close.diff().eq(0.0).astype("float64").where(close.notna())
    return flat.rolling(window, min_periods=mp).mean()


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Corwin-Schultz (2012) two-period high-low effective spread estimator.

    Reported as a diagnostic only. On this universe it collapses to roughly zero for
    thin names because Yahoo's hourly high-low ranges are degenerate, so it is never
    used to charge costs. Kept so the report can show *why* it was rejected.
    """
    hi, lo = high.where(high > 0), low.where(low > 0)
    hl = np.log(hi / lo) ** 2
    beta = hl + hl.shift(1)
    h2 = pd.concat([hi, hi.shift(1)]).groupby(level=0).max()
    l2 = pd.concat([lo, lo.shift(1)]).groupby(level=0).min()
    gamma = np.log(h2 / l2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).replace([np.inf, -np.inf], np.nan)


def parkinson_vol(high: pd.DataFrame, low: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Parkinson high-low volatility estimator, more efficient than close-to-close."""
    mp = min_periods if min_periods is not None else max(2, window // 4)
    hl = np.log(high.where(high > 0) / low.where(low > 0)) ** 2
    return np.sqrt(hl.rolling(window, min_periods=mp).mean() / (4.0 * np.log(2.0)))
