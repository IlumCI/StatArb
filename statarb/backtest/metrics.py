"""Performance metrics with honest inference.

Two problems make naive statistics badly misleading on this strategy, and both are
handled here.

First, trades are *correlated across assets*: fourteen followers reacting to the same
leader move in the same hour is one bet, not fourteen, so per-trade standard errors
must be clustered by timestamp. Ignoring this overstates t-statistics by roughly the
square root of the cross-section width.

Second, many configurations get tried during research, so the best observed Sharpe is
biased upward by selection. The deflated Sharpe ratio corrects for that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PerformanceStats:
    n_bars: int
    n_trades: int
    gross_return: float
    net_return: float
    gross_sharpe: float
    net_sharpe: float
    net_sharpe_ci: tuple[float, float]
    deflated_sharpe: float
    hit_rate: float
    mean_trade_bp: float
    mean_trade_bp_se: float
    mean_trade_t: float
    turnover: float
    max_drawdown: float
    cost_bp_per_trade: float
    breakeven_cost_bp: float

    def to_dict(self) -> dict:
        return asdict(self)


def sharpe(returns: pd.Series, bars_per_year: int) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def max_drawdown(returns: pd.Series) -> float:
    equity = returns.fillna(0.0).cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def clustered_mean_se(values: pd.Series, clusters: pd.Index | pd.Series) -> tuple[float, float, int]:
    """Mean and cluster-robust standard error of per-trade returns.

    Clusters are timestamps: all trades placed on the same bar share the same leader
    shock and are therefore treated as one observation for inference purposes.
    """
    df = pd.DataFrame({"v": values.to_numpy()}, index=pd.Index(clusters, name="cluster")).dropna()
    if df.empty:
        return float("nan"), float("nan"), 0
    per_cluster = df.groupby(level="cluster")["v"].mean()
    g = len(per_cluster)
    if g < 2:
        return float(per_cluster.mean()), float("nan"), g
    # Weighting every cluster equally is the conservative choice: a bar with more
    # simultaneous trades carries no more independent information than a bar with one.
    return float(per_cluster.mean()), float(per_cluster.std(ddof=1) / np.sqrt(g)), g


def stationary_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap resample indices.

    Geometric block lengths preserve the autocorrelation of the return series, which
    matters because hourly strategy returns are far from independent.
    """
    p = 1.0 / max(block_size, 1)
    idx = np.empty(n, dtype=np.int64)
    cur = rng.integers(0, n)
    for i in range(n):
        idx[i] = cur
        cur = rng.integers(0, n) if rng.random() < p else (cur + 1) % n
    return idx


def bootstrap_sharpe_ci(
    returns: pd.Series,
    bars_per_year: int,
    *,
    block_size: int = 168,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 11,
) -> tuple[float, float]:
    r = returns.dropna().to_numpy()
    if len(r) < block_size * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    scale = np.sqrt(bars_per_year)
    stats_out = np.empty(n_boot)
    for b in range(n_boot):
        sample = r[stationary_bootstrap_indices(len(r), block_size, rng)]
        sd = sample.std()
        stats_out[b] = sample.mean() / sd * scale if sd > 0 else np.nan
    lo, hi = np.nanpercentile(stats_out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def deflated_sharpe_ratio(
    observed_sharpe: float, returns: pd.Series, n_trials: int, bars_per_year: int
) -> float:
    """Bailey & Lopez de Prado deflated Sharpe ratio.

    Returns the probability that the true Sharpe is positive, after correcting for the
    number of configurations tried and for skew/kurtosis in the return distribution.
    A value below ~0.95 means the result is not distinguishable from selection luck.
    """
    r = returns.dropna()
    n = len(r)
    if n < 30 or not np.isfinite(observed_sharpe):
        return float("nan")
    sr = observed_sharpe / np.sqrt(bars_per_year)  # per-bar Sharpe
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    # Expected maximum Sharpe under the null of no skill across n_trials attempts.
    emc = 0.5772156649
    trials = max(int(n_trials), 2)
    z1 = stats.norm.ppf(1.0 - 1.0 / trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (trials * np.e))
    sr0 = (1 - emc) * z1 + emc * z2
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0 / np.sqrt(n)) * np.sqrt(n - 1) / denom))


def summarise(
    bar_returns_gross: pd.Series,
    bar_returns_net: pd.Series,
    trade_returns_bp: pd.Series,
    trade_clusters: pd.Series,
    turnover: pd.Series,
    cost_bp_per_trade: float,
    *,
    bars_per_year: int,
    block_size: int,
    n_boot: int,
    n_trials: int,
    seed: int,
) -> PerformanceStats:
    mean_bp, se_bp, _ = clustered_mean_se(trade_returns_bp, trade_clusters)
    t_stat = mean_bp / se_bp if se_bp and np.isfinite(se_bp) and se_bp > 0 else float("nan")
    net_sr = sharpe(bar_returns_net, bars_per_year)
    return PerformanceStats(
        n_bars=int(bar_returns_net.notna().sum()),
        n_trades=int(trade_returns_bp.notna().sum()),
        gross_return=float(bar_returns_gross.sum()),
        net_return=float(bar_returns_net.sum()),
        gross_sharpe=sharpe(bar_returns_gross, bars_per_year),
        net_sharpe=net_sr,
        net_sharpe_ci=bootstrap_sharpe_ci(
            bar_returns_net, bars_per_year, block_size=block_size, n_boot=n_boot, seed=seed
        ),
        deflated_sharpe=deflated_sharpe_ratio(net_sr, bar_returns_net, n_trials, bars_per_year),
        hit_rate=float((trade_returns_bp.dropna() > 0).mean()) if trade_returns_bp.notna().any() else float("nan"),
        mean_trade_bp=mean_bp,
        mean_trade_bp_se=se_bp,
        mean_trade_t=t_stat,
        turnover=float(turnover.sum()),
        max_drawdown=max_drawdown(bar_returns_net),
        cost_bp_per_trade=cost_bp_per_trade,
        breakeven_cost_bp=float(mean_bp) if np.isfinite(mean_bp) else float("nan"),
    )
