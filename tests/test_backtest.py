"""Walk-forward, metrics and end-to-end backtest behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.backtest.engine import run_backtest
from statarb.backtest.metrics import (
    clustered_mean_se,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe,
)
from statarb.backtest.walkforward import assert_no_overlap, coverage, walk_forward_folds


def test_folds_are_purged_and_ordered():
    folds = walk_forward_folds(10000, 2000, 500, 24, holding_bars=2)
    assert folds
    assert_no_overlap(folds, 2, 24)
    for f in folds:
        assert f.train_end <= f.test_start
        assert f.test_start - f.train_end >= 26
    # Test windows must tile without overlapping each other.
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end <= b.test_start


def test_insufficient_purge_is_detected():
    from statarb.backtest.walkforward import Fold

    bad = [Fold(0, 0, 1000, 1001, 1500)]
    with pytest.raises(AssertionError, match="purge"):
        assert_no_overlap(bad, holding_bars=2, embargo_bars=24)


def test_coverage_is_fraction_of_sample():
    folds = walk_forward_folds(10000, 2000, 500, 24, holding_bars=1)
    assert 0.0 < coverage(folds, 10000) <= 1.0


def test_clustered_se_exceeds_naive_se_when_trades_are_correlated():
    """Correlated same-bar trades must not be counted as independent observations."""
    rng = np.random.default_rng(0)
    n_bars, per_bar = 200, 10
    shocks = rng.normal(0, 1.0, n_bars)
    values, clusters = [], []
    for i, s in enumerate(shocks):
        for _ in range(per_bar):
            values.append(s + rng.normal(0, 0.05))  # nearly identical within a bar
            clusters.append(i)
    v = pd.Series(values)
    _, se_clustered, n_clusters = clustered_mean_se(v, pd.Index(clusters))
    se_naive = v.std(ddof=1) / np.sqrt(len(v))
    assert n_clusters == n_bars
    assert se_clustered > se_naive * 2


def test_sharpe_and_drawdown():
    r = pd.Series([0.01] * 100)
    assert sharpe(r, 8760) > 0
    assert max_drawdown(r) == pytest.approx(0.0)
    down = pd.Series([0.01, -0.05, 0.01])
    assert max_drawdown(down) < 0


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 5000))
    sr = sharpe(r, 8760)
    few = deflated_sharpe_ratio(sr, r, n_trials=2, bars_per_year=8760)
    many = deflated_sharpe_ratio(sr, r, n_trials=1000, bars_per_year=8760)
    assert many < few


def test_end_to_end_backtest_runs(synthetic_panel, fast_config):
    cfg = fast_config.with_overrides(
        portfolio=type(fast_config.portfolio)(
            **{**fast_config.portfolio.__dict__, "construction": "both"}
        )
    )
    results = run_backtest(synthetic_panel, cfg)
    assert set(results) == {"cross_sectional", "directional"}
    for name, r in results.items():
        assert r.stats.n_trades > 0, f"{name} placed no trades"
        assert np.isfinite(r.stats.mean_trade_bp)
        # Net must never beat gross: costs are non-negative by construction.
        assert r.stats.net_return <= r.stats.gross_return + 1e-9
        assert r.weights.abs().sum(axis=1).max() <= 1.0 + 1e-6


def test_predictions_are_out_of_sample_only(synthetic_panel, fast_config):
    """No prediction may exist before the first fold's test window opens."""
    from statarb.backtest.engine import prepare

    prep = prepare(synthetic_panel, fast_config)
    first_test_ts = prep.dataset.index[prep.folds[0].test_start]
    ts = prep.predictions.dropna().index.get_level_values("timestamp")
    assert ts.min() >= first_test_ts
