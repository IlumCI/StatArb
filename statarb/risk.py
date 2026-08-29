"""Risk layer: the Hawkes crash filter and position scaling.

The exploratory work found that the lead-lag edge inverts in the extreme tail, and
that splitting on follower jump intensity separates a strongly positive regime from a
strongly negative one. This module turns that into two mechanisms:

* a **hard gate** that refuses to trade an asset whose conditional jump intensity or
  branching ratio is too high, and
* a **soft scaler** that shrinks position size smoothly as intensity rises.

Both are computed causally: parameters are refit on trailing windows only, and the
intensity at bar ``t`` uses strictly earlier jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config import HawkesConfig
from statarb.models.hawkes import (
    HawkesParams,
    conditional_intensity,
    fit_hawkes,
    jump_events,
)


@dataclass(frozen=True)
class HawkesState:
    """Per-bar, per-asset crash-risk view."""

    intensity: pd.DataFrame        # lambda(t)
    intensity_ratio: pd.DataFrame  # lambda(t) / mu, i.e. excitation above baseline
    branching: pd.DataFrame        # alpha/beta as of the most recent refit
    params: dict[str, list[tuple[pd.Timestamp, HawkesParams]]]

    def tradable(self, cfg: HawkesConfig) -> pd.DataFrame:
        """Hard gate: True where the asset is calm enough to trade."""
        calm = self.intensity_ratio <= cfg.max_intensity_ratio
        stable = self.branching <= cfg.max_branching_ratio
        # An asset with no fitted parameters yet is not tradable: absence of a risk
        # estimate is not evidence of low risk.
        return (calm & stable).fillna(False)

    def size_scalar(self, cfg: HawkesConfig) -> pd.DataFrame:
        """Soft scaler in (0, 1]: shrinks size as excitation rises above baseline."""
        excess = (self.intensity_ratio - 1.0).clip(lower=0.0)
        return (1.0 / (1.0 + cfg.soft_kappa * excess)).fillna(0.0)


def trailing_jump_threshold(returns: pd.DataFrame, cfg: HawkesConfig) -> pd.DataFrame:
    """Trailing |return| quantile, shifted one bar.

    Full-sample thresholds would leak the future distribution of shocks into the
    definition of what counted as a jump at each historical bar.
    """
    return (
        returns.abs()
        .rolling(cfg.jump_window, min_periods=cfg.jump_min_periods)
        .quantile(cfg.jump_quantile)
        .shift(1)
    )


def rolling_hawkes_state(
    returns: pd.DataFrame, cfg: HawkesConfig, *, negative_only: bool = True
) -> HawkesState:
    """Fit Hawkes parameters on a rolling basis and evaluate intensity every bar.

    Parameters are refit every ``refit_every`` bars on the trailing ``fit_window``,
    then held fixed while intensity is evaluated forward. This mirrors what a live
    system can do: you cannot refit on data you have not seen.
    """
    index = returns.index
    n = len(index)
    thresholds = trailing_jump_threshold(returns, cfg)

    intensity = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    ratio = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    branching = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    history: dict[str, list[tuple[pd.Timestamp, HawkesParams]]] = {}

    # Refit points start once there is a full fit window of trailing data.
    starts = list(range(cfg.fit_window, n, cfg.refit_every))
    if not starts:
        return HawkesState(intensity, ratio, branching, history)

    for sym in returns.columns:
        events_all = jump_events(returns[sym], thresholds[sym], negative_only=negative_only)
        fits: list[tuple[pd.Timestamp, HawkesParams]] = []
        for start in starts:
            lo = max(0, start - cfg.fit_window)
            # Only events strictly before the refit bar may inform the fit.
            window_events = events_all[(events_all >= lo) & (events_all < start)] - lo
            params = fit_hawkes(
                window_events, float(start - lo),
                mu_bounds=cfg.mu_bounds, alpha_bounds=cfg.alpha_bounds, beta_bounds=cfg.beta_bounds,
            )
            fits.append((index[start], params))

            stop = min(start + cfg.refit_every, n)
            eval_positions = np.arange(start, stop, dtype="float64")
            # Evaluate against the full event history, not just the fit window, so a
            # cascade that began before the refit still registers.
            past = events_all[events_all < stop].astype("float64")
            lam = conditional_intensity(params, past, eval_positions)
            intensity.iloc[start:stop, intensity.columns.get_loc(sym)] = lam
            ratio.iloc[start:stop, ratio.columns.get_loc(sym)] = lam / max(params.mu, 1e-12)
            branching.iloc[start:stop, branching.columns.get_loc(sym)] = params.branching_ratio
        history[sym] = fits

    return HawkesState(intensity, ratio, branching, history)


def combine_masks(*masks: pd.DataFrame) -> pd.DataFrame:
    """Logical AND across tradability masks, treating missing as not tradable."""
    out: pd.DataFrame | None = None
    for m in masks:
        filled = m.fillna(False).astype(bool)
        out = filled if out is None else (out & filled.reindex_like(out).fillna(False))
    if out is None:
        raise ValueError("no masks given")
    return out


def liquidity_mask(dollar_volume: pd.DataFrame, min_dollar_volume: float) -> pd.DataFrame:
    """Refuse to trade an asset with too little volume to absorb an order."""
    return (dollar_volume >= min_dollar_volume).fillna(False)


def capacity_weight_cap(
    dollar_volume: pd.DataFrame,
    gross_notional: float,
    max_participation: float,
) -> pd.DataFrame:
    """Largest weight each asset can carry without exceeding its participation limit.

    Preferred over a binary mask: rather than dropping an illiquid name outright, the
    book holds as much of it as the volume supports. A name whose cap rounds to zero
    is excluded naturally, and one with room keeps a proportionally smaller position.
    """
    if gross_notional <= 0:
        raise ValueError("gross_notional must be positive")
    cap = (dollar_volume * max_participation) / gross_notional
    return cap.clip(lower=0.0).fillna(0.0)


def apply_capacity(
    weights: pd.DataFrame,
    cap: pd.DataFrame,
    *,
    renormalise: bool = True,
    min_tradable_weight: float = 0.01,
) -> pd.DataFrame:
    """Clip weights to their capacity cap, optionally restoring gross exposure.

    Renormalising pushes the freed notional into names that still have room, which is
    what a real allocator does. It cannot resurrect a bar where nothing is tradable:
    those stay flat.
    """
    # A name whose capacity cannot support even a minimum ticket is dropped outright
    # rather than held in vanishing size, so trade counts reflect what would really
    # be sent to the venue.
    usable = cap.where(cap >= min_tradable_weight, 0.0)
    capped = weights.clip(lower=-usable, upper=usable, axis=None)
    capped = capped.where(usable > 0, 0.0)
    if not renormalise:
        return capped
    target = weights.abs().sum(axis=1)
    actual = capped.abs().sum(axis=1)
    scale = (target / actual.where(actual > 0)).clip(upper=10.0).fillna(0.0)
    rescaled = capped.mul(scale, axis=0)
    # Rescaling must not reintroduce a breach, so clip once more and accept the
    # shortfall in gross exposure when capacity genuinely runs out.
    return rescaled.clip(lower=-usable, upper=usable, axis=None)


def capacity_mask(
    dollar_volume: pd.DataFrame,
    gross_notional: float,
    max_weight: float,
    max_participation: float,
) -> pd.DataFrame:
    """Refuse assets where even a minimum-size position would swamp the bar's volume.

    This is the constraint that separates a paper edge from a tradable one. The
    strongest lead-lag signals in this universe sit on assets turning over a few
    dollars an hour; without a capacity check the backtest happily "fills" orders
    thousands of times larger than anything that trades, and the resulting equity
    curve is fiction.
    """
    position = gross_notional * max_weight
    capacity = dollar_volume * max_participation
    return (capacity >= position).fillna(False)
