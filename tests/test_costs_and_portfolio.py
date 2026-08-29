"""Cost model and portfolio construction invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.config import CostConfig, PortfolioConfig
from statarb.costs import breakeven_cost_bp, cost_breakdown, half_spread
from statarb.portfolio.cross_sectional import CrossSectionalLongShort
from statarb.portfolio.directional import DirectionalBetaHedged
from statarb.risk import apply_capacity, capacity_weight_cap


@pytest.fixture
def dvol():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    return pd.DataFrame(
        {"LIQ": 1e7, "MID": 1e5, "THIN": 1e3, "DUST": 1e0}, index=idx, dtype="float64"
    )


def test_illiquid_assets_are_charged_more(dvol):
    hs = half_spread(dvol, CostConfig())
    assert hs["LIQ"].iloc[0] < hs["MID"].iloc[0] <= hs["THIN"].iloc[0] <= hs["DUST"].iloc[0]


def test_cost_increases_with_order_size(dvol):
    cfg = CostConfig()
    small = cost_breakdown(pd.DataFrame(100.0, index=dvol.index, columns=dvol.columns), dvol, cfg)
    large = cost_breakdown(pd.DataFrame(10000.0, index=dvol.index, columns=dvol.columns), dvol, cfg)
    assert (large.total_bp >= small.total_bp).all().all()
    assert large.total_bp["MID"].iloc[0] > small.total_bp["MID"].iloc[0]


def test_missing_liquidity_is_charged_the_cap_not_zero(dvol):
    """Unknown liquidity must be treated as expensive, never as free."""
    cfg = CostConfig()
    unknown = dvol.copy()
    unknown["MID"] = np.nan
    bd = cost_breakdown(pd.DataFrame(1000.0, index=dvol.index, columns=dvol.columns), unknown, cfg)
    assert bd.impact_bp["MID"].iloc[0] == pytest.approx(cfg.max_impact_bp)
    assert bd.half_spread_bp["MID"].iloc[0] == cfg.tier_half_spread_bp[-1]


def test_breakeven_cost_matches_gross_edge():
    assert breakeven_cost_bp(10e-4, turnover_per_trade=2.0) == pytest.approx(10.0)


def test_cross_sectional_book_is_dollar_neutral():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    cols = [f"A{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(rng.normal(size=(5, 8)), index=idx, columns=cols)
    tradable = pd.DataFrame(True, index=idx, columns=cols)
    w = CrossSectionalLongShort(PortfolioConfig()).weights(scores, tradable)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9)   # dollar neutral
    assert np.allclose(w.abs().sum(axis=1), 1.0, atol=1e-9)  # fully invested


def test_constructions_respect_tradable_mask():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    cols = ["A", "B", "C", "D", "E", "F"]
    scores = pd.DataFrame(1.0, index=idx, columns=cols)
    scores["A"] = 5.0
    tradable = pd.DataFrame(True, index=idx, columns=cols)
    tradable["A"] = False  # the most attractive name is gated out
    for ctor in (CrossSectionalLongShort, DirectionalBetaHedged):
        w = ctor(PortfolioConfig()).weights(scores, tradable)
        assert (w["A"].abs() < 1e-12).all(), f"{ctor.__name__} traded a gated asset"


def test_capacity_cap_zeroes_untradeable_names():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    dv = pd.DataFrame({"BIG": 1e8, "DUST": 1.0}, index=idx)
    cap = capacity_weight_cap(dv, gross_notional=1e5, max_participation=0.1)
    w = pd.DataFrame({"BIG": 0.5, "DUST": 0.5}, index=idx)
    capped = apply_capacity(w, cap)
    assert (capped["DUST"].abs() < 1e-12).all()
    assert (capped["BIG"].abs() > 0).all()


def test_capacity_never_exceeds_participation_limit():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    dv = pd.DataFrame({"A": 5e4, "B": 1e8}, index=idx)
    gross, part = 1e5, 0.1
    cap = capacity_weight_cap(dv, gross, part)
    w = pd.DataFrame({"A": 0.9, "B": 0.1}, index=idx)
    capped = apply_capacity(w, cap)
    notional = capped.abs() * gross
    assert (notional <= dv * part + 1e-6).all().all()
