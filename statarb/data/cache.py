"""Parquet-backed bar cache.

Yahoo rate-limits hard and only serves a rolling 730d window of hourly bars, so the
cache is not just a speed-up: it is how history is retained past the point where the
API stops serving it, and how a backtest stays reproducible between runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from statarb.data.yahoo import FetchSpec, fetch

log = logging.getLogger(__name__)


def cache_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_")
    return Path(cache_dir) / f"{safe}__{interval}.parquet"


def read_cached(cache_dir: Path, symbol: str, interval: str) -> pd.DataFrame | None:
    path = cache_path(cache_dir, symbol, interval)
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt cache is rare
        log.warning("unreadable cache %s (%s), refetching", path, exc)
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "timestamp"
    return frame.sort_index()


def write_cache(cache_dir: Path, symbol: str, interval: str, frame: pd.DataFrame) -> Path:
    path = cache_path(cache_dir, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_index().to_parquet(path)
    return path


def merge_bars(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union two bar frames, preferring freshly fetched rows on overlap."""
    if old is None or old.empty:
        return new.sort_index()
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def load_symbol(
    symbol: str,
    *,
    interval: str,
    lookback: str,
    cache_dir: Path,
    backend: str = "chart",
    refresh: bool = True,
    **fetch_kwargs: object,
) -> pd.DataFrame:
    """Return bars for one symbol, extending the cache from the API when asked."""
    cached = read_cached(cache_dir, symbol, interval)
    if not refresh and cached is not None and not cached.empty:
        return cached

    spec = FetchSpec(symbol=symbol, interval=interval, lookback=lookback)
    fresh = fetch(spec, backend=backend, **fetch_kwargs)
    merged = merge_bars(cached, fresh)
    write_cache(cache_dir, symbol, interval, merged)
    return merged


def load_universe(
    symbols: tuple[str, ...] | list[str],
    *,
    interval: str,
    lookback: str,
    cache_dir: Path,
    backend: str = "chart",
    refresh: bool = True,
    **fetch_kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Load every symbol, tolerating individual failures.

    One dead ticker must not abort a run; the caller decides whether the surviving
    cross-section is wide enough to trade.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = load_symbol(
                sym,
                interval=interval,
                lookback=lookback,
                cache_dir=cache_dir,
                backend=backend,
                refresh=refresh,
                **fetch_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately tolerant
            cached = read_cached(cache_dir, sym, interval)
            if cached is not None and not cached.empty:
                log.warning("%s: fetch failed (%s), using cache", sym, exc)
                out[sym] = cached
            else:
                log.error("%s: unavailable (%s), dropping from universe", sym, exc)
    return out
