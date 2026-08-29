"""Transaction cost model.

This is the module that decides whether the strategy is real. The measured gross edge
is roughly 10-19bp per trade, and a round trip on the illiquid tail can easily cost
more than that, so costs are modelled conservatively and reported prominently rather
than tucked into a footnote.

Corwin-Schultz is deliberately *not* used for charging. On this universe it estimates
near-zero spreads for the thinnest names because Yahoo's hourly high-low ranges are
degenerate, which would flatter the backtest exactly where reality is worst. It is
computed in ``features.micro`` as a diagnostic so the report can show why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config import CostConfig

BP = 1e-4


@dataclass(frozen=True)
class CostBreakdown:
    """Per-side costs in basis points, plus the notional they were charged on."""

    half_spread_bp: pd.DataFrame
    fee_bp: pd.DataFrame
    impact_bp: pd.DataFrame
    total_bp: pd.DataFrame

    def mean_total_bp(self) -> pd.Series:
        return self.total_bp.mean(axis=0, skipna=True)


def tier_from_dollar_volume(dollar_volume: pd.DataFrame, n_tiers: int) -> pd.DataFrame:
    """Cross-sectional liquidity tier per bar, 0 = most liquid.

    Uses the trailing dollar-volume estimate, so tier membership is knowable at the
    decision bar and shifts as relative liquidity changes.
    """
    pct = dollar_volume.rank(axis=1, pct=True, na_option="keep")
    # floor, not ceil-minus-one: the latter collapses the top two tiers together, so
    # the most liquid asset and the second most liquid get charged the same spread.
    tiers = np.floor((1.0 - pct) * n_tiers)
    return tiers.clip(lower=0, upper=n_tiers - 1)


def half_spread(dollar_volume: pd.DataFrame, cfg: CostConfig) -> pd.DataFrame:
    """Assumed effective half-spread in bp, from the asset's liquidity tier."""
    tiers = tier_from_dollar_volume(dollar_volume, len(cfg.tier_half_spread_bp))
    lookup = np.asarray(cfg.tier_half_spread_bp, dtype="float64")
    idx = tiers.to_numpy()
    out = np.full(idx.shape, np.nan)
    valid = np.isfinite(idx)
    out[valid] = lookup[idx[valid].astype(int)]
    # An asset with no liquidity estimate is assumed to be the worst tier, never the best.
    out[~valid] = lookup[-1]
    return pd.DataFrame(out, index=dollar_volume.index, columns=dollar_volume.columns)


def impact(
    trade_notional: pd.DataFrame, dollar_volume: pd.DataFrame, cfg: CostConfig
) -> pd.DataFrame:
    """Amihud-style market impact in bp, proportional to participation rate.

    Capped so that one thin bar cannot dominate the backtest, but the cap is high
    enough that trading a meaningful fraction of an asset's volume still hurts.
    """
    participation = trade_notional.abs().div(dollar_volume.where(dollar_volume > 0))
    raw = cfg.impact_coef_bp * participation.replace([np.inf, -np.inf], np.nan)
    # Missing liquidity means unknown impact; charge the cap rather than nothing.
    return raw.fillna(cfg.max_impact_bp).clip(upper=cfg.max_impact_bp)


def cost_breakdown(
    trade_notional: pd.DataFrame, dollar_volume: pd.DataFrame, cfg: CostConfig
) -> CostBreakdown:
    """Per-side cost in bp for a set of trades."""
    hs = half_spread(dollar_volume, cfg)
    fee = pd.DataFrame(cfg.taker_fee_bp, index=hs.index, columns=hs.columns)
    imp = impact(trade_notional, dollar_volume, cfg)
    return CostBreakdown(hs, fee, imp, hs + fee + imp)


def charge(
    side_notional: pd.DataFrame, dollar_volume: pd.DataFrame, cfg: CostConfig
) -> pd.DataFrame:
    """Cost in currency for one side of a trade. Sign-free: trading always costs.

    ``side_notional`` is the notional of a single fill, not the round trip. Impact
    scales with the size of each individual order, so passing doubled notional here
    would overstate impact by charging it against a trade that never happens at once.
    """
    bd = cost_breakdown(side_notional, dollar_volume, cfg)
    return side_notional.abs() * bd.total_bp * BP


def round_trip_bp(dollar_volume: pd.DataFrame, cfg: CostConfig, participation: float = 0.0) -> pd.DataFrame:
    """Round-trip cost in bp at a given participation rate. Reporting helper."""
    hs = half_spread(dollar_volume, cfg)
    imp = min(cfg.impact_coef_bp * participation, cfg.max_impact_bp)
    return 2.0 * (hs + cfg.taker_fee_bp + imp)


def breakeven_cost_bp(gross_return_per_trade: float, turnover_per_trade: float = 2.0) -> float:
    """Round-trip cost in bp at which a given gross edge nets to exactly zero.

    This is the headline viability number: compare it against the round-trip cost the
    venue would actually charge. If breakeven sits below realistic cost, the strategy
    does not work no matter how good the forecast looks.
    """
    if turnover_per_trade <= 0:
        return float("nan")
    return float(gross_return_per_trade / BP / turnover_per_trade * 2.0)
