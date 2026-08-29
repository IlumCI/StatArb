"""Lead-lag detection and signal construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.features.leadlag import cross_correlation, gate_mask
from statarb.features.returns import log_returns


def test_detects_injected_lag(synthetic_panel):
    """Followers built as a lagged copy of the leader must be detected as such."""
    lr = log_returns(synthetic_panel.close)
    table = cross_correlation(lr[synthetic_panel.leader], lr[list(synthetic_panel.followers)], max_lag=3)
    # FOLL0..2 and FOLL4 were constructed with a lag; FOLL3 is contemporaneous.
    assert table.loc["FOLL0-USD", "lag1"] > 0.2
    assert table.loc["FOLL2-USD", "lag2"] > 0.15
    assert table.loc["FOLL3-USD", "lag0"] > 0.5
    # Reverse causality must stay near zero: followers do not lead the leader.
    assert table["reverse_lag1"].abs().max() < 0.1


def test_gate_fires_at_expected_rate():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.01, 5000), index=pd.date_range("2024-01-01", periods=5000, freq="h", tz="UTC"))
    mask = gate_mask(s, window=1000, quantile=0.9, min_periods=500)
    rate = mask[mask.notna()].mean()
    # Roughly the complement of the quantile, allowing for the warm-up period.
    assert 0.05 < rate < 0.15


def test_gate_uses_only_trailing_data():
    """A huge future move must not change whether an earlier bar was gated."""
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 0.01, 3000), index=pd.date_range("2024-01-01", periods=3000, freq="h", tz="UTC"))
    bumped = base.copy()
    bumped.iloc[2000:] *= 50
    a = gate_mask(base, 500, 0.9, 200).iloc[:2000]
    b = gate_mask(bumped, 500, 0.9, 200).iloc[:2000]
    pd.testing.assert_series_equal(a, b)
