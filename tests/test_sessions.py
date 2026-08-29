"""Session structure and bar-grid detection.

Both behaviours here were live bugs. Equity bars sit at :30 past the hour, which a
floor-to-the-hour grid check discarded wholesale; and a parquet round trip can return
millisecond-resolution timestamps, which broke an offset computed as if they were
nanoseconds. Each cost a silently near-empty panel, so both are pinned down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.data.panel import build_panel, drop_partial_bar
from statarb.data.sessions import (
    bars_remaining_in_session,
    is_session_end,
    is_session_start,
    mask_cross_session,
    session_ids,
    session_summary,
    tradable_entry_mask,
)


def equity_index(n_sessions: int = 4) -> pd.DatetimeIndex:
    """US equity hourly bars: 7 per session, on the :30 grid."""
    stamps = []
    for day in pd.bdate_range("2024-03-04", periods=n_sessions):
        for hour in range(14, 21):
            stamps.append(pd.Timestamp(f"{day:%Y-%m-%d} {hour}:30", tz="UTC"))
    return pd.DatetimeIndex(stamps)


def crypto_index(n: int = 100) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def _frame(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": np.arange(1.0, n + 1), "volume": 1e5},
        index=idx,
    )


def test_grid_detection_keeps_half_past_equity_bars():
    """Equity bars on the :30 grid must all survive; only the ragged bar is dropped."""
    idx = equity_index()
    frame = _frame(idx)
    kept = drop_partial_bar(frame, "1h")
    assert len(kept) == len(idx)

    # Append Yahoo's in-progress bar at an arbitrary wall-clock second.
    partial = frame.copy()
    stamp = pd.Timestamp("2024-03-07 20:47:13", tz="UTC")
    partial.loc[stamp] = partial.iloc[-1]
    assert len(drop_partial_bar(partial.sort_index(), "1h")) == len(idx)


def test_grid_detection_keeps_on_the_hour_crypto_bars():
    idx = crypto_index()
    assert len(drop_partial_bar(_frame(idx), "1h")) == len(idx)


def test_grid_detection_survives_millisecond_timestamps():
    """A parquet round trip yields ms resolution; the offset maths must still work."""
    idx = equity_index()
    frame = _frame(idx)
    frame.index = pd.DatetimeIndex(frame.index).astype("datetime64[ms, UTC]")
    assert len(drop_partial_bar(frame, "1h")) == len(idx)


def test_continuous_market_is_one_session():
    s = session_ids(crypto_index(), tz=None)
    assert s.nunique() == 1
    assert is_session_start(s).sum() == 1
    # Nothing is ever untradable for session reasons in a continuous market.
    assert tradable_entry_mask(s, 1, 2).iloc[:-3].all()


def test_equity_sessions_split_by_exchange_date():
    idx = equity_index(4)
    s = session_ids(idx, tz="America/New_York")
    assert s.nunique() == 4
    assert is_session_start(s).sum() == 4
    assert is_session_end(s).sum() == 4
    assert session_summary(s)["bars_per_session_median"] == 7.0


def test_overnight_returns_are_masked():
    """A close-to-close return spanning a session boundary must be dropped."""
    idx = equity_index(3)
    s = session_ids(idx, tz="America/New_York")
    prices = pd.DataFrame({"A": np.arange(1.0, len(idx) + 1)}, index=idx)
    masked = mask_cross_session(prices.diff(), s)
    # 3 session starts: first bar of each session has no valid intraday predecessor.
    assert masked["A"].isna().sum() == 3
    assert masked["A"].iloc[1] == pytest.approx(1.0)


def test_entry_mask_keeps_round_trip_inside_the_session():
    idx = equity_index(2)
    s = session_ids(idx, tz="America/New_York")
    mask = tradable_entry_mask(s, entry_lag=1, holding_bars=2)
    per_session = mask.groupby(s.to_numpy()).sum()
    # 7 bars, needing 2 bars after the signal, leaves 5 tradable entries per session.
    assert (per_session == 5).all()
    assert bars_remaining_in_session(s).max() == 6


def test_panel_carries_sessions_through():
    idx = equity_index(5)
    bars = {s: _frame(idx) for s in ("IWM", "AAA", "BBB")}
    panel = build_panel(
        bars, leader="IWM", followers=("AAA", "BBB"), session_tz="America/New_York"
    )
    assert panel.sessions.nunique() == 5
    assert panel.slice(start=idx[7]).sessions.nunique() == 4
