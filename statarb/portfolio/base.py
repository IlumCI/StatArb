"""Portfolio construction interface.

Both constructions map a per-asset score at bar ``t`` to target weights held over the
next holding window. Keeping them behind one interface is what lets the backtest run
them on identical signals and costs, so the comparison is like for like.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from statarb.config import PortfolioConfig


class PortfolioConstructor(ABC):
    """Maps scores to target weights, as a fraction of gross notional."""

    name: str = "base"

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg

    @abstractmethod
    def weights(self, scores: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
        """Target weight per (bar, asset). Rows sum to at most 1.0 gross.

        ``tradable`` masks assets the risk layer has gated out for that bar.
        """

    def _clip(self, w: pd.DataFrame) -> pd.DataFrame:
        return w.clip(lower=-self.cfg.max_weight, upper=self.cfg.max_weight)

    @staticmethod
    def _normalise_gross(w: pd.DataFrame) -> pd.DataFrame:
        """Scale each bar so gross exposure is exactly 1, leaving all-flat bars flat."""
        gross = w.abs().sum(axis=1)
        return w.div(gross.where(gross > 0), axis=0).fillna(0.0)


def leader_hedge_weight(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """Leader weight that neutralises the basket's aggregate beta.

    Returned as the weight to hold in the leader, i.e. the negative of the portfolio's
    net beta exposure.
    """
    return -(weights * betas.reindex_like(weights)).sum(axis=1)


def apply_vol_target(
    weights: pd.DataFrame,
    portfolio_returns: pd.Series,
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    """Scale weights toward a volatility target using trailing realised vol.

    The scalar is shifted by one bar so that sizing at ``t`` uses only volatility
    observed strictly before ``t``.
    """
    if cfg.vol_target_annual is None:
        return weights
    mp = max(2, cfg.vol_window // 4)
    realised = portfolio_returns.rolling(cfg.vol_window, min_periods=mp).std()
    annualised = realised * np.sqrt(cfg.bars_per_year)
    scalar = (cfg.vol_target_annual / annualised.replace(0.0, np.nan)).shift(1)
    scalar = scalar.clip(upper=cfg.max_leverage).fillna(1.0)
    return weights.mul(scalar, axis=0)
