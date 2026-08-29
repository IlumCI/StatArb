"""Event-driven walk-forward backtest.

Execution model, stated explicitly because it is where lead-lag backtests usually go
wrong: a signal formed at the **close of bar t** is executed at the **open of bar
t+1** and held for ``holding_bars``. Filling at the signal bar's own close would be
lookahead, and on stale illiquid names it is the single easiest way to manufacture a
Sharpe ratio that cannot be traded.

Costs are charged on every entry and exit, in bp of traded notional, using the
liquidity-tiered model in :mod:`statarb.costs`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.backtest.metrics import PerformanceStats, summarise
from statarb.backtest.walkforward import assert_no_overlap, coverage, walk_forward_folds
from statarb.config import Config
from statarb.costs import BP, cost_breakdown
from statarb.data.panel import Panel
from statarb.features.dataset import TARGET_COLUMN, Dataset, build_dataset
from statarb.features.returns import log_returns
from statarb.models.baseline import LeadLagBaseline
from statarb.portfolio.base import PortfolioConstructor, apply_vol_target
from statarb.portfolio.cross_sectional import CrossSectionalLongShort
from statarb.portfolio.directional import DirectionalBetaHedged
from statarb.risk import (
    HawkesState,
    apply_capacity,
    capacity_weight_cap,
    combine_masks,
    rolling_hawkes_state,
)

log = logging.getLogger(__name__)

CONSTRUCTORS: dict[str, type[PortfolioConstructor]] = {
    "cross_sectional": CrossSectionalLongShort,
    "directional": DirectionalBetaHedged,
}


@dataclass
class BacktestResult:
    name: str
    stats: PerformanceStats
    bar_returns_gross: pd.Series
    bar_returns_net: pd.Series
    weights: pd.DataFrame
    trade_returns_bp: pd.Series
    trade_clusters: pd.Series
    costs_bp: pd.Series
    predictions: pd.Series
    diagnostics: dict = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return self.bar_returns_net.fillna(0.0).cumsum()


@dataclass
class Prepared:
    """Everything derived from the panel that the backtest needs, computed once."""

    panel: Panel
    dataset: Dataset
    hawkes: HawkesState
    predictions: pd.Series
    folds: list
    baseline_coefs: pd.DataFrame


def _predict_walk_forward(dataset: Dataset, cfg: Config, model_factory=None) -> tuple[pd.Series, pd.DataFrame]:
    """Fit on each fold's training window and predict its test window.

    Returns predictions covering only out-of-sample bars; in-sample bars stay NaN and
    are excluded from every performance statistic downstream.
    """
    frame = dataset.frame
    index = dataset.index
    folds = walk_forward_folds(
        len(index), cfg.backtest.train_bars, cfg.backtest.test_bars,
        cfg.backtest.embargo_bars, cfg.signal.holding_bars,
    )
    assert_no_overlap(folds, cfg.signal.holding_bars, cfg.backtest.embargo_bars)

    ts = frame.index.get_level_values("timestamp")
    preds = pd.Series(np.nan, index=frame.index, dtype="float64")
    coef_frames = []
    for fold in folds:
        train_hi = index[fold.train_end - 1]
        train_lo = index[fold.train_start]
        test_lo, test_hi = index[fold.test_start], index[fold.test_end - 1]
        train = frame[(ts >= train_lo) & (ts <= train_hi)]
        test = frame[(ts >= test_lo) & (ts <= test_hi)]
        train = train.dropna(subset=[TARGET_COLUMN])
        if train.empty or test.empty:
            continue
        model = (model_factory or LeadLagBaseline)()
        model.fit(train)
        preds.loc[test.index] = model.predict(test).to_numpy()
        c = model.coefficient_table()
        c["fold"] = fold.index
        coef_frames.append(c)
    coefs = pd.concat(coef_frames) if coef_frames else pd.DataFrame()
    return preds, coefs


def prepare(panel: Panel, cfg: Config, *, model_factory=None) -> Prepared:
    """Build features, fit the risk model and produce out-of-sample predictions."""
    dataset = build_dataset(panel, cfg.signal)
    follower_returns = log_returns(panel.close[list(panel.followers)])
    log.info("fitting rolling Hawkes state (%d assets)", len(panel.followers))
    hawkes = rolling_hawkes_state(follower_returns, cfg.hawkes)
    log.info("running walk-forward predictions")
    preds, coefs = _predict_walk_forward(dataset, cfg, model_factory)
    folds = walk_forward_folds(
        len(dataset.index), cfg.backtest.train_bars, cfg.backtest.test_bars,
        cfg.backtest.embargo_bars, cfg.signal.holding_bars,
    )
    return Prepared(panel, dataset, hawkes, preds, folds, coefs)


def run_construction(
    prep: Prepared,
    cfg: Config,
    construction: str,
    *,
    use_hawkes_gate: bool = True,
    use_hawkes_scaler: bool = True,
    use_capacity_gate: bool = True,
    name: str | None = None,
) -> BacktestResult:
    """Turn predictions into positions, apply costs and summarise."""
    ds, panel = prep.dataset, prep.panel
    followers = list(panel.followers)
    index = ds.index

    scores = prep.predictions.unstack(level="symbol").reindex(columns=followers).reindex(index)

    # Tradability: only on gated leader-move bars, and only for assets the risk layer
    # is willing to hold.
    gate_rows = ds.gate.reindex(index).fillna(False)
    tradable = pd.DataFrame(
        np.repeat(gate_rows.to_numpy()[:, None], len(followers), axis=1),
        index=index, columns=followers,
    )
    tradable &= scores.notna()
    if use_hawkes_gate:
        tradable = combine_masks(tradable, prep.hawkes.tradable(cfg.hawkes).reindex_like(tradable))


    ctor = CONSTRUCTORS[construction](cfg.portfolio)
    weights = ctor.weights(scores, tradable)
    if use_hawkes_scaler:
        weights = weights * prep.hawkes.size_scalar(cfg.hawkes).reindex_like(weights).fillna(0.0)
    if use_capacity_gate:
        # Constrain each position to what the asset's volume can absorb. This is the
        # step that decides whether a measured edge is actually harvestable.
        cap = capacity_weight_cap(
            ds.dollar_volume.reindex_like(weights),
            cfg.portfolio.gross_notional,
            cfg.portfolio.max_participation,
        )
        weights = apply_capacity(weights, cap)

    # Realised return on the position opened at t+1 open and closed holding_bars later.
    realised = ds.wide(TARGET_COLUMN).reindex(index).reindex(columns=followers)

    gross_by_asset = weights * realised
    gross = gross_by_asset.sum(axis=1, min_count=1)

    # Every position is opened and closed within the holding window, so each bar's
    # traded notional is twice the weight taken. This deliberately assumes no netting
    # between consecutive bars: overlapping positions in an illiquid name would not
    # net at zero cost in practice.
    # Impact scales with the size of each individual order, so it is charged against
    # the per-side notional and then doubled for the round trip, rather than being
    # charged once against a fictional double-size order.
    notional = weights.abs() * cfg.portfolio.gross_notional
    bd = cost_breakdown(notional, ds.dollar_volume.reindex_like(weights), cfg.costs)
    cost_frac = 2.0 * (notional * bd.total_bp * BP) / cfg.portfolio.gross_notional
    cost_by_bar = cost_frac.sum(axis=1, min_count=1)
    trade_notional = notional * 2.0

    net = gross - cost_by_bar.fillna(0.0)
    net = apply_vol_target(net.to_frame("r"), net, cfg.portfolio)["r"]

    # Per-trade statistics, clustered by bar for inference.
    active = weights.abs() > 1e-12
    trade_ret = (np.sign(weights) * realised).where(active)
    trade_long = trade_ret.stack(future_stack=True).dropna() / BP
    clusters = pd.Series(
        trade_long.index.get_level_values("timestamp"), index=trade_long.index, name="cluster"
    )
    per_trade_cost = bd.total_bp.where(active).stack(future_stack=True).dropna()

    stats = summarise(
        gross, net, trade_long, clusters,
        turnover=trade_notional.sum(axis=1) / cfg.portfolio.gross_notional,
        cost_bp_per_trade=float(per_trade_cost.mean() * 2.0) if len(per_trade_cost) else float("nan"),
        bars_per_year=cfg.portfolio.bars_per_year,
        block_size=cfg.backtest.block_bootstrap_size,
        n_boot=cfg.backtest.n_bootstrap,
        n_trials=cfg.backtest.n_trials_for_deflation,
        seed=cfg.backtest.seed,
    )
    return BacktestResult(
        name=name or construction,
        stats=stats,
        bar_returns_gross=gross,
        bar_returns_net=net,
        weights=weights,
        trade_returns_bp=trade_long,
        trade_clusters=clusters,
        costs_bp=per_trade_cost,
        predictions=prep.predictions,
        diagnostics={
            "oos_coverage": coverage(prep.folds, len(index)),
            "n_folds": len(prep.folds),
            "bars_traded": int((weights.abs().sum(axis=1) > 0).sum()),
            "hawkes_gate": use_hawkes_gate,
            "hawkes_scaler": use_hawkes_scaler,
            "capacity_gate": use_capacity_gate,
        },
    )


def run_backtest(panel: Panel, cfg: Config, *, model_factory=None) -> dict[str, BacktestResult]:
    """Run every requested construction on one shared set of predictions."""
    prep = prepare(panel, cfg, model_factory=model_factory)
    which = (
        list(CONSTRUCTORS)
        if cfg.portfolio.construction == "both"
        else [cfg.portfolio.construction]
    )
    return {c: run_construction(prep, cfg, c) for c in which}
