"""Timestamp-aligned OHLCV panel across the universe.

Every downstream component consumes a :class:`Panel`. Aligning once, here, is what
makes the "no lookahead" guarantee checkable: all frames share one monotonic index,
and nothing downstream ever reindexes or forward-fills across it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")

_INTERVAL_TO_OFFSET = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "1h", "90m": "90min", "1h": "1h", "1d": "1D", "1wk": "1W",
}


@dataclass(frozen=True)
class Panel:
    """Aligned bars for a leader plus followers.

    Attributes are ``timestamp x symbol`` frames. ``followers`` excludes the leader,
    which is kept separate because it plays a structurally different role.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    leader: str
    followers: tuple[str, ...]
    interval: str

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index  # type: ignore[return-value]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.close.columns)

    def field(self, name: str) -> pd.DataFrame:
        return getattr(self, name)

    def follower_frame(self, name: str) -> pd.DataFrame:
        return self.field(name)[list(self.followers)]

    def leader_series(self, name: str) -> pd.Series:
        return self.field(name)[self.leader]

    def slice(self, start=None, end=None) -> Panel:
        """Restrict to a time window, keeping every field consistent."""
        idx = self.index
        mask = np.ones(len(idx), dtype=bool)
        if start is not None:
            mask &= idx >= pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else idx <= pd.Timestamp(end)
        return Panel(
            open=self.open[mask], high=self.high[mask], low=self.low[mask],
            close=self.close[mask], volume=self.volume[mask],
            leader=self.leader, followers=self.followers, interval=self.interval,
        )

    def iloc(self, start: int, stop: int) -> Panel:
        return Panel(
            open=self.open.iloc[start:stop], high=self.high.iloc[start:stop],
            low=self.low.iloc[start:stop], close=self.close.iloc[start:stop],
            volume=self.volume.iloc[start:stop],
            leader=self.leader, followers=self.followers, interval=self.interval,
        )

    def describe(self) -> pd.DataFrame:
        """Per-symbol coverage summary, useful for spotting a broken feed."""
        # Mean, not median: Yahoo reports zero volume on roughly half of hourly crypto
        # bars, so the median dollar volume of every asset is identically zero.
        stale = staleness(self.close)
        return pd.DataFrame(
            {
                "bars": self.close.notna().sum(),
                "first": self.close.apply(lambda s: s.first_valid_index()),
                "last": self.close.apply(lambda s: s.last_valid_index()),
                "zero_return_frac": stale.mean(),
                "mean_dollar_vol": (self.close * self.volume).mean(),
            }
        )


def drop_partial_bar(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop bars not aligned to the interval grid.

    Yahoo appends an in-progress bar stamped at the current wall-clock second. Trading
    on it would mean acting on a partial bar, so it is removed by construction.
    """
    offset = _INTERVAL_TO_OFFSET.get(interval)
    if offset is None or frame.empty:
        return frame
    idx = pd.DatetimeIndex(frame.index)
    aligned = idx.floor(offset) == idx
    return frame[aligned]


def staleness(close: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame marking bars whose close did not move.

    A repeated print in an illiquid name usually means nothing traded. This is the
    honest read on how much of the measured lead-lag is a real, tradable move versus
    a stale quote that will never fill.
    """
    return close.diff().eq(0.0) & close.notna()


def build_panel(
    bars: dict[str, pd.DataFrame],
    *,
    leader: str,
    followers: tuple[str, ...] | list[str],
    interval: str = "1h",
    min_cross_section_coverage: float = 0.6,
) -> Panel:
    """Align per-symbol bar frames onto one shared index.

    Bars are intersected on the leader's timeline (no leader move means no signal) and
    then filtered to timestamps where enough of the cross-section actually traded.
    """
    if leader not in bars:
        raise ValueError(f"leader {leader!r} missing from fetched bars")
    present = [s for s in followers if s in bars and not bars[s].empty]
    if not present:
        raise ValueError("no followers available")

    cleaned = {s: drop_partial_bar(bars[s], interval) for s in [leader, *present]}
    cleaned = {s: f for s, f in cleaned.items() if not f.empty}
    if leader not in cleaned:
        raise ValueError(f"leader {leader!r} had no complete bars")
    present = [s for s in present if s in cleaned]

    index = cleaned[leader].index
    frames: dict[str, pd.DataFrame] = {}
    for fld in FIELDS:
        cols = {s: cleaned[s][fld].reindex(index) for s in [leader, *present]}
        frames[fld] = pd.DataFrame(cols, index=index).astype("float64")

    # Require both a live leader print and a wide enough cross-section to rank.
    follower_close = frames["close"][present]
    coverage = follower_close.notna().mean(axis=1)
    keep = frames["close"][leader].notna() & (coverage >= min_cross_section_coverage)
    frames = {k: v[keep] for k, v in frames.items()}

    return Panel(
        open=frames["open"], high=frames["high"], low=frames["low"],
        close=frames["close"], volume=frames["volume"].fillna(0.0),
        leader=leader, followers=tuple(present), interval=interval,
    )
