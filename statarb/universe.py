"""The traded universe: one liquid leader and a basket of illiquid followers.

Follower selection is deliberately restricted to Solana-ecosystem tokens. The whole
lead-lag premise needs a *shared information source* (SOL) plus a genuine liquidity
ladder, and this basket spans roughly $50M down to $0.1M in average dollar volume.
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


@dataclass(frozen=True)
class Universe:
    leader: str
    followers: tuple[str, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.leader, *self.followers)

    def __len__(self) -> int:
        return len(self.followers)


def default_universe() -> Universe:
    return Universe(leader=LEADER, followers=FOLLOWERS)


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
