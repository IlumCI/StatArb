"""Directional, beta-hedged construction.

Every follower takes the leader's direction, sized by its predicted catch-up, and the
basket's aggregate beta is hedged with an offsetting leader position. This is what the
exploratory validation measured. It needs only one cheap, liquid hedge leg instead of
shorting the illiquid tail, but it remains a factor bet: it profits when followers
catch up to the leader, and the hedge only removes the *average* exposure.
"""

from __future__ import annotations

import pandas as pd

from statarb.portfolio.base import PortfolioConstructor


class DirectionalBetaHedged(PortfolioConstructor):
    name = "directional"

    def weights(self, scores: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
        masked = scores.where(tradable & scores.notna())
        # Size by conviction, but cap the influence of any single extreme score.
        conviction = masked.clip(
            lower=-masked.abs().stack().quantile(0.99) if masked.notna().any().any() else None,
            upper=masked.abs().stack().quantile(0.99) if masked.notna().any().any() else None,
        )
        w = conviction.fillna(0.0)
        w = self._normalise_gross(w)
        return self._clip(w)
