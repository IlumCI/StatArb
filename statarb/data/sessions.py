"""Trading-session structure.

Crypto trades continuously, so consecutive bars are genuinely consecutive and a
return between any two of them is a return you could have traded. Equities are not
like that: the bar at 16:00 and the bar at 09:30 the next morning are separated by an
overnight auction, a night of news, and no ability to trade.

Two consequences drive everything in this module:

1. A return computed across a session boundary is an **overnight gap**, not an
   intraday move. Feeding it to a lead-lag model teaches the model that the leader
   predicts an overnight jump it can never trade into.
2. A position whose entry or exit falls in a different session from its signal is
   held overnight, taking gap risk the strategy is not being paid for.

The system therefore treats sessions as hard walls: returns are nulled across them,
and trades are only opened when signal, entry and exit all land in the same session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_ids(index: pd.DatetimeIndex, *, tz: str | None = None) -> pd.Series:
    """Label each bar with the trading session it belongs to.

    ``tz`` is the exchange timezone: sessions are then calendar dates in that zone,
    which is the correct definition and is immune to daylight saving shifting bar
    timestamps around in UTC. ``tz=None`` means a continuously traded market, where
    the whole sample is one unbroken session and no boundary logic applies.
    """
    idx = pd.DatetimeIndex(index)
    if tz is None:
        return pd.Series(np.zeros(len(idx), dtype="int64"), index=idx, name="session")
    local = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz)
    return pd.Series(pd.Index(local.date).astype("datetime64[ns]"), index=idx, name="session")


def is_session_start(sessions: pd.Series) -> pd.Series:
    """True for the first bar of each session."""
    return sessions.ne(sessions.shift(1)) | sessions.shift(1).isna()


def is_session_end(sessions: pd.Series) -> pd.Series:
    """True for the last bar of each session."""
    return sessions.ne(sessions.shift(-1)) | sessions.shift(-1).isna()


def bar_of_session(sessions: pd.Series) -> pd.Series:
    """0-based position of each bar within its session."""
    return sessions.groupby(sessions).cumcount()


def bars_remaining_in_session(sessions: pd.Series) -> pd.Series:
    """Number of bars strictly after this one within the same session."""
    counts = sessions.map(sessions.value_counts())
    return counts - bar_of_session(sessions) - 1


def mask_cross_session(frame: pd.DataFrame, sessions: pd.Series, periods: int = 1) -> pd.DataFrame:
    """Null rows whose ``periods``-bar lookback crosses into an earlier session.

    Applied to any differenced quantity so that overnight gaps never enter a feature.
    """
    same = sessions.eq(sessions.shift(periods))
    return frame.where(same.reindex(frame.index).fillna(False), other=np.nan)


def tradable_entry_mask(sessions: pd.Series, entry_lag: int, holding_bars: int) -> pd.Series:
    """True where a signal at this bar can be entered and exited in the same session.

    A signal at bar ``t`` buys at ``t + entry_lag`` and sells ``holding_bars - 1``
    bars later, so the last bar it touches is ``t + entry_lag + holding_bars - 1``.
    Requiring that many bars to remain keeps the whole round trip intraday.
    """
    needed = entry_lag + holding_bars - 1
    return bars_remaining_in_session(sessions) >= needed


def session_summary(sessions: pd.Series) -> dict:
    counts = sessions.value_counts()
    return {
        "n_sessions": int(sessions.nunique()),
        "n_bars": int(len(sessions)),
        "bars_per_session_median": float(counts.median()),
        "bars_per_session_min": int(counts.min()),
        "bars_per_session_max": int(counts.max()),
    }
