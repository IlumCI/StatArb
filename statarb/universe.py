"""Traded universes: one liquid leader and a basket of illiquid followers.

The lead-lag premise needs two things: a *shared information source* the whole basket
responds to, and a genuine liquidity ladder beneath it. Two markets are provided.

**Solana** — SOL-USD leads its ecosystem tokens. Continuously traded, a very wide
liquidity ladder, but the thinnest names turn over a few dollars an hour.

**US equities** — IWM (Russell 2000) leads a basket of small and micro caps. Session
based rather than continuous, a much narrower liquidity ladder, but every name in it
trades real volume. The follower list was screened on liquidity and data coverage
alone, never on measured lead-lag, so the basket is not selected on the thing being
tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEADER = "SOL-USD"

#: Followers verified to have full 730d hourly history on Yahoo Finance.
FOLLOWERS: tuple[str, ...] = (
    "JTO-USD",      # Jito
    "RENDER-USD",   # Render
    "ORCA-USD",     # Orca
    "RAY-USD",      # Raydium
    "PYTH-USD",     # Pyth
    "WIF-USD",      # dogwifhat
    "HNT-USD",      # Helium
    "W-USD",        # Wormhole
    "TNSR-USD",     # Tensor
    "KMNO-USD",     # Kamino
    "FIDA-USD",     # Bonfida
    "MOBILE-USD",   # Helium Mobile
    "SAMO-USD",     # Samoyedcoin
    "ATLAS-USD",    # Star Atlas
)

#: Number of ADV-based liquidity tiers. Tier 0 is the most liquid.
N_TIERS = 4


#: US small/micro caps surviving a signal-blind screen of a broader candidate list:
#: at least 4,500 hourly bars of history and average dollar volume per bar between
#: $30k and $3m. Selection never referenced the lead-lag coefficient being measured.
EQUITY_LEADER = "IWM"

EQUITY_FOLLOWERS: tuple[str, ...] = (
    "TALO", "WLDN", "KOPN", "CEVA", "DAKT", "NVEC", "MYE", "SD", "IRMD",
    "REI", "AMPY", "RICK", "JAKK", "LTRX", "FMBH", "NNBR", "ESOA", "ELMD",
    "FLXS", "EPM", "RCMT", "LAKE", "INTT", "UUU", "PKOH", "GENC", "INVE",
)


@dataclass(frozen=True)
class Universe:
    leader: str
    followers: tuple[str, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.leader, *self.followers)

    def __len__(self) -> int:
        return len(self.followers)


@dataclass(frozen=True)
class Market:
    """A tradable universe plus the venue conventions that go with it."""

    name: str
    leader: str
    followers: tuple[str, ...]
    #: Exchange timezone defining session boundaries; None for a continuous market.
    session_tz: str | None
    #: Bars in a trading year, used to annualise Sharpe ratios.
    bars_per_year: int
    description: str

    @property
    def universe(self) -> Universe:
        return Universe(leader=self.leader, followers=self.followers)

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.leader, *self.followers)

    @property
    def is_continuous(self) -> bool:
        return self.session_tz is None


SOLANA_MARKET = Market(
    name="solana",
    leader=LEADER,
    followers=FOLLOWERS,
    session_tz=None,
    bars_per_year=24 * 365,
    description="SOL-USD leading Solana ecosystem tokens, traded continuously",
)

EQUITY_MARKET = Market(
    name="equity",
    leader=EQUITY_LEADER,
    followers=EQUITY_FOLLOWERS,
    session_tz="America/New_York",
    # 7 hourly bars per US equity session (the last one is a half bar), 252 sessions.
    bars_per_year=7 * 252,
    description="IWM leading US small and micro caps, 7 bars per session",
)

MARKETS: dict[str, Market] = {m.name: m for m in (SOLANA_MARKET, EQUITY_MARKET)}


def get_market(name: str) -> Market:
    if name not in MARKETS:
        raise ValueError(f"unknown market {name!r}; choose from {sorted(MARKETS)}")
    return MARKETS[name]


def default_universe(market: str = "solana") -> Universe:
    return get_market(market).universe


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Per-bar traded notional in USD."""
    return close.mul(volume).astype("float64")


def average_dollar_volume(
    close: pd.DataFrame, volume: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    """Trailing average dollar volume per bar.

    Trailing rather than full-sample: tier membership must be knowable at decision time.

    Yahoo reports zero volume on roughly half of all hourly crypto bars, so any
    per-bar liquidity measure is unusable and only a window average carries signal.
    This is the single liquidity estimate the cost model and tiering are allowed to use.
    """
    dv = dollar_volume(close, volume)
    mp = min_periods if min_periods is not None else max(1, window // 4)
    return dv.rolling(window, min_periods=mp).mean()


def liquidity_tiers(adv: pd.DataFrame, n_tiers: int = N_TIERS) -> pd.DataFrame:
    """Assign each (bar, asset) an integer liquidity tier, 0 = most liquid.

    Ranking is cross-sectional within each bar, so tiers use only information available
    at that bar and adapt as relative liquidity shifts over the sample.
    """
    if adv.empty:
        return adv.copy()
    # rank(pct=True) gives 0..1 ascending, so invert to make 0 the most liquid.
    # floor rather than ceil-minus-one, which would merge the top two tiers.
    pct = adv.rank(axis=1, pct=True, na_option="keep")
    tiers = np.floor((1.0 - pct) * n_tiers)
    return tiers.clip(lower=0, upper=n_tiers - 1)


def static_tiers(adv: pd.DataFrame, n_tiers: int = N_TIERS) -> pd.Series:
    """One tier per asset from mean ADV over the sample. Reporting only.

    This *is* full-sample and therefore must never feed the cost model or any signal;
    it exists so summary tables can label assets consistently.
    """
    med = adv.mean(axis=0, skipna=True)
    ranks = med.rank(ascending=False, na_option="keep", method="first")
    tiers = np.floor((ranks - 1) / max(len(med) / n_tiers, 1e-9))
    return tiers.clip(lower=0, upper=n_tiers - 1).fillna(n_tiers - 1).astype(int)
