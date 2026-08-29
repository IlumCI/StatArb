"""The structural guarantee: no feature may depend on the future.

This is the single most important test in the suite. A lead-lag strategy on stale,
illiquid prices is trivially easy to make profitable by accident, and almost every way
of doing so reduces to a feature peeking one bar ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.data.panel import build_panel
from statarb.features.dataset import FEATURE_COLUMNS, build_dataset


def test_features_unchanged_when_future_is_corrupted(synthetic_bars, fast_config):
    """Corrupt every bar after a cut point; features up to it must not move.

    If any feature reads the future, scrambling the future changes it and this fails.
    """
    cut = 2000
    clean = build_panel(
        synthetic_bars, leader="LEAD-USD",
        followers=tuple(s for s in synthetic_bars if s != "LEAD-USD"), interval="1h",
    )
    corrupted_bars = {}
    rng = np.random.default_rng(0)
    for sym, frame in synthetic_bars.items():
        f = frame.copy()
        tail = f.index[cut:]
        # Replace the entire future with different, wildly scaled data.
        f.loc[tail, ["open", "high", "low", "close"]] *= rng.uniform(2.0, 5.0, size=(len(tail), 4))
        f.loc[tail, "volume"] *= 100.0
        corrupted_bars[sym] = f
    dirty = build_panel(
        corrupted_bars, leader="LEAD-USD",
        followers=tuple(s for s in synthetic_bars if s != "LEAD-USD"), interval="1h",
    )

    ds_clean = build_dataset(clean, fast_config.signal)
    ds_dirty = build_dataset(dirty, fast_config.signal)

    cutoff_ts = clean.index[cut - 1]
    for ds in (ds_clean, ds_dirty):
        assert ds.frame.index.names == ["timestamp", "symbol"]

    a = ds_clean.features[ds_clean.features.index.get_level_values("timestamp") <= cutoff_ts]
    b = ds_dirty.features[ds_dirty.features.index.get_level_values("timestamp") <= cutoff_ts]
    assert len(a) == len(b) and len(a) > 0

    for col in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            a[col], b[col], check_names=False, rtol=1e-9, atol=1e-12,
            obj=f"feature {col!r} leaks future information",
        )


def test_gate_is_unchanged_by_future(synthetic_bars, fast_config):
    """The trailing quantile gate must also be blind to the future."""
    cut = 2000
    followers = tuple(s for s in synthetic_bars if s != "LEAD-USD")
    clean = build_panel(synthetic_bars, leader="LEAD-USD", followers=followers, interval="1h")
    bumped = {s: f.copy() for s, f in synthetic_bars.items()}
    lead = bumped["LEAD-USD"]
    lead.loc[lead.index[cut:], ["open", "high", "low", "close"]] *= 10.0
    dirty = build_panel(bumped, leader="LEAD-USD", followers=followers, interval="1h")

    g_clean = build_dataset(clean, fast_config.signal).gate.iloc[:cut]
    g_dirty = build_dataset(dirty, fast_config.signal).gate.iloc[:cut]
    pd.testing.assert_series_equal(g_clean, g_dirty, check_names=False)


def test_forward_return_is_strictly_forward(synthetic_panel, fast_config):
    """The target must be built from bars strictly after the decision bar."""
    from statarb.features.returns import forward_return

    followers = list(synthetic_panel.followers)
    fwd = forward_return(
        synthetic_panel.open[followers], synthetic_panel.close[followers],
        entry_lag=1, holding_bars=1,
    )
    sym = followers[0]
    i = 500
    expected = np.log(
        synthetic_panel.close[sym].iloc[i + 1] / synthetic_panel.open[sym].iloc[i + 1]
    )
    assert np.isclose(fwd[sym].iloc[i], expected)
    # The last bars cannot have a resolved target.
    assert np.isnan(fwd[sym].iloc[-1])


def test_entry_lag_zero_is_rejected():
    """Filling at the signal bar's own close is lookahead and must be refused."""
    import pytest

    from statarb.features.returns import forward_return

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="lookahead"):
        forward_return(df, df, entry_lag=0)
