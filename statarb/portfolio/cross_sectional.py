"""Cross-sectional long/short construction.

Ranks followers by predicted catch-up and takes the top quantile long against the
bottom quantile short in equal notional. This is market neutral by construction: the
common SOL move cancels between the legs, so what remains is the *relative* catch-up
prediction. The cost is that both legs pay the spread, and the short leg on the
illiquid tail may not be borrowable at all in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.portfolio.base import PortfolioConstructor


class CrossSectionalLongShort(PortfolioConstructor):
    name = "cross_sectional"

    def weights(self, scores: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
        masked = scores.where(tradable & scores.notna())
        n_valid = masked.notna().sum(axis=1)
        # Ranking needs enough names on both sides to be meaningful.
        enough = n_valid >= max(2 * self.cfg.min_names_per_side, 4)

        ranks = masked.rank(axis=1, pct=True, na_option="keep")
        q = self.cfg.quantile
        long_leg = (ranks >= 1.0 - q).astype("float64").where(masked.notna(), 0.0)
        short_leg = (ranks <= q).astype("float64").where(masked.notna(), 0.0)

        n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
        n_short = short_leg.sum(axis=1).replace(0.0, np.nan)
        # Half the gross on each side, so the book is dollar neutral bar by bar.
        w = long_leg.div(n_long, axis=0) * 0.5 - short_leg.div(n_short, axis=0) * 0.5
        w = w.where(enough, 0.0).fillna(0.0)
        return self._clip(w)
