"""Yahoo Finance bar client.

Two backends. ``chart`` talks to the public v8 chart endpoint directly through urllib,
which has no third-party dependency and is what was verified working in this
environment. ``yfinance`` is available for users who prefer it, selected via config.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
#: Keep this short. A full browser UA string sent from a non-browser TLS stack gets
#: fingerprinted by Yahoo and answered with 429; a bare "Mozilla/5.0" is served fine.
_UA = "Mozilla/5.0"

OHLCV = ("open", "high", "low", "close", "volume")


class DataUnavailable(RuntimeError):
    """Raised when a symbol cannot be fetched after retries."""


@dataclass(frozen=True)
class FetchSpec:
    symbol: str
    interval: str = "1h"
    lookback: str = "730d"


#: Yahoo rate-limits aggressively. Serialise requests with a minimum spacing so a
#: full-universe fetch does not trip 429 on the first few symbols.
_MIN_REQUEST_INTERVAL_S = 1.5
_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _get_json(url: str, timeout: float) -> dict:
    _throttle()
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        return json.load(resp)


def fetch_chart(
    spec: FetchSpec,
    *,
    max_retries: int = 4,
    backoff_s: float = 2.0,
    timeout_s: float = 40.0,
) -> pd.DataFrame:
    """Fetch OHLCV bars for one symbol, indexed by tz-aware UTC timestamp.

    Retries with exponential backoff on transport errors, which the public endpoint
    produces intermittently under load.
    """
    url = CHART_URL.format(symbol=spec.symbol) + f"?range={spec.lookback}&interval={spec.interval}"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            payload = _get_json(url, timeout_s)
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
            if attempt == max_retries - 1:
                raise DataUnavailable(f"{spec.symbol}: {exc}") from exc
            # 429 means we are being shed deliberately, so back off far harder than
            # for a transport blip and honour Retry-After when the server sends one.
            throttled = isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 503)
            if throttled:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    sleep_s = float(retry_after) if retry_after else 0.0
                except (TypeError, ValueError):
                    sleep_s = 0.0
                sleep_s = max(sleep_s, backoff_s * 5 * (2**attempt))
            else:
                sleep_s = backoff_s * (2**attempt)
            log.warning("fetch %s failed (%s), retry in %.0fs", spec.symbol, exc, sleep_s)
            time.sleep(sleep_s)
    else:  # pragma: no cover - loop always breaks or raises
        raise DataUnavailable(f"{spec.symbol}: {last_err}")

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise DataUnavailable(f"{spec.symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise DataUnavailable(f"{spec.symbol}: empty result")

    res = results[0]
    ts = res.get("timestamp")
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    if not ts or not quote:
        raise DataUnavailable(f"{spec.symbol}: no quote data")

    # Build from arrays, not Series: passing Series alongside an explicit index makes
    # pandas align on the Series' own RangeIndex and silently yields an all-NaN frame.
    index = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(ts, dtype="int64"), unit="s", utc=True), name="timestamp"
    )
    frame = pd.DataFrame(
        {
            k: np.asarray(
                [np.nan if v is None else v for v in (quote.get(k) or [None] * len(index))],
                dtype="float64",
            )
            for k in OHLCV
        },
        index=index,
    )
    # A bar with no close never traded; keep the row out entirely rather than
    # forward-filling, so downstream staleness detection sees the real gap.
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    # Yahoo occasionally emits a partial in-progress bar with null volume.
    frame["volume"] = frame["volume"].fillna(0.0)
    for col in ("open", "high", "low"):
        frame[col] = frame[col].where(frame[col] > 0, frame["close"])
    return frame.astype("float64")


def fetch_yfinance(spec: FetchSpec, **_: object) -> pd.DataFrame:
    """Alternate backend using the yfinance package, if installed."""
    try:
        import yfinance  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataUnavailable("yfinance backend requested but yfinance is not installed") from exc

    raw = yfinance.Ticker(spec.symbol).history(
        period=spec.lookback, interval=spec.interval, auto_adjust=False
    )
    if raw is None or raw.empty:
        raise DataUnavailable(f"{spec.symbol}: yfinance returned no rows")
    frame = raw.rename(columns=str.lower)[list(OHLCV)].astype("float64")
    idx = pd.to_datetime(frame.index, utc=True)
    frame.index = pd.DatetimeIndex(idx, name="timestamp")
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch(spec: FetchSpec, backend: str = "chart", **kwargs: object) -> pd.DataFrame:
    if backend == "yfinance":
        return fetch_yfinance(spec, **kwargs)
    if backend == "chart":
        return fetch_chart(spec, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown backend {backend!r}")


def synthetic_bars(
    n: int, start: str = "2024-01-01", freq: str = "h", seed: int = 0, drift: float = 0.0
) -> pd.DataFrame:
    """Deterministic fake bars. Used by tests so they never touch the network."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name="timestamp")
    open_ = np.concatenate([[close[0]], close[:-1]])
    noise = np.abs(rng.normal(0, 0.002, size=n)) * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + noise,
            "low": np.minimum(open_, close) - noise,
            "close": close,
            "volume": rng.lognormal(10, 1, size=n),
        },
        index=idx,
    )
