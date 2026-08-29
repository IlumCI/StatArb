"""Shared synthetic fixtures. No test touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.config import (
    BacktestConfig,
    Config,
    HawkesConfig,
    PortfolioConfig,
    SignalConfig,
)
from statarb.data.panel import build_panel


def make_leader(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": rng.lognormal(12, 0.5, n),
        },
        index=idx,
    )


def make_follower(leader: pd.DataFrame, lag: int, beta: float, noise: float, seed: int, volume_scale: float = 1.0):
    """Follower that mechanically reprices the leader's move ``lag`` bars later."""
    rng = np.random.default_rng(seed)
    lead_ret = np.log(leader["close"] / leader["open"]).to_numpy()
    n = len(lead_ret)
    own = np.zeros(n)
    own[lag:] = beta * lead_ret[:-lag] if lag else beta * lead_ret
    own = own + rng.normal(0, noise, n)
    close = 10 * np.exp(np.cumsum(own))
    open_ = np.concatenate([[10.0], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": rng.lognormal(12, 0.5, n) * volume_scale,
        },
        index=leader.index,
    )


@pytest.fixture
def synthetic_bars() -> dict[str, pd.DataFrame]:
    leader = make_leader(3000, seed=1)
    bars = {"LEAD-USD": leader}
    for i, (lag, beta, noise) in enumerate(
        [(1, 0.8, 0.004), (1, 0.6, 0.006), (2, 0.5, 0.005), (0, 0.9, 0.004), (1, 0.7, 0.005)]
    ):
        bars[f"FOLL{i}-USD"] = make_follower(leader, lag, beta, noise, seed=100 + i)
    return bars


@pytest.fixture
def synthetic_panel(synthetic_bars):
    return build_panel(
        synthetic_bars,
        leader="LEAD-USD",
        followers=tuple(s for s in synthetic_bars if s != "LEAD-USD"),
        interval="1h",
    )


@pytest.fixture
def fast_config() -> Config:
    """Small windows so tests run in seconds rather than minutes."""
    return Config(
        signal=SignalConfig(
            beta_window=200, beta_min_periods=60, gate_window=300,
            gate_min_periods=100, gate_quantile=0.8, holding_bars=1,
        ),
        hawkes=HawkesConfig(
            jump_window=300, jump_min_periods=100, refit_every=400, fit_window=800,
        ),
        portfolio=PortfolioConfig(gross_notional=10_000, construction="cross_sectional"),
        backtest=BacktestConfig(train_bars=800, test_bars=300, embargo_bars=5, n_bootstrap=100),
    )
