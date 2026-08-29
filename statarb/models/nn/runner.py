"""Walk-forward driver for the neural overlay.

Mirrors the baseline's fold structure exactly so the two are directly comparable: the
same folds, the same purge, the same test bars. The only difference is that the net
persists across folds and keeps adapting, which is the whole point of an online model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.backtest.walkforward import walk_forward_folds
from statarb.config import Config
from statarb.features.dataset import FEATURE_COLUMNS, TARGET_COLUMN, Dataset
from statarb.models.baseline import LeadLagBaseline
from statarb.models.nn.blend import BlendState, blend
from statarb.models.nn.dataset import GATE_COLUMNS, build_matrices
from statarb.models.nn.net import build_net, torch_available
from statarb.models.nn.online import OnlineTrainer

log = logging.getLogger(__name__)


@dataclass
class OverlayResult:
    baseline_predictions: pd.Series
    blended_predictions: pd.Series
    nn_residual_predictions: pd.Series
    ic_history: list[tuple[pd.Timestamp, float, float]] = field(default_factory=list)
    regime_shifts: int = 0
    trained: bool = False

    def ic_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.ic_history, columns=["timestamp", "ic", "nn_weight"]).set_index(
            "timestamp"
        )


def run_overlay(dataset: Dataset, cfg: Config, hawkes_ratio: pd.Series | None = None) -> OverlayResult:
    """Fit baseline and neural overlay fold by fold, adapting the net online.

    Returns both the baseline-only predictions and the blended ones, so every
    downstream comparison is like for like.
    """
    frame = dataset.frame
    index = dataset.index
    folds = walk_forward_folds(
        len(index), cfg.backtest.train_bars, cfg.backtest.test_bars,
        cfg.backtest.embargo_bars, cfg.signal.holding_bars,
    )
    ts = frame.index.get_level_values("timestamp")

    baseline_preds = pd.Series(np.nan, index=frame.index, dtype="float64")
    blended_preds = pd.Series(np.nan, index=frame.index, dtype="float64")
    nn_preds = pd.Series(np.nan, index=frame.index, dtype="float64")

    use_nn = cfg.nn.enabled and torch_available()
    if cfg.nn.enabled and not use_nn:
        log.warning("torch unavailable; running baseline only")

    trainer: OnlineTrainer | None = None
    blend_state = BlendState(
        min_baseline_weight=cfg.nn.min_baseline_weight, ic_window=cfg.nn.ic_window
    )
    result = OverlayResult(baseline_preds, blended_preds, nn_preds, trained=use_nn)

    if use_nn:
        import torch  # noqa: PLC0415

        torch.manual_seed(cfg.nn.seed)
        np.random.seed(cfg.nn.seed)

    for fold in folds:
        train_lo, train_hi = index[fold.train_start], index[fold.train_end - 1]
        test_lo, test_hi = index[fold.test_start], index[fold.test_end - 1]
        train = frame[(ts >= train_lo) & (ts <= train_hi)].dropna(subset=[TARGET_COLUMN])
        test = frame[(ts >= test_lo) & (ts <= test_hi)]
        if train.empty or test.empty:
            continue

        model = LeadLagBaseline()
        model.fit(train)
        base_train = model.predict(train)
        base_test = model.predict(test)
        baseline_preds.loc[test.index] = base_test.to_numpy()
        blended_preds.loc[test.index] = base_test.to_numpy()

        if not use_nn:
            continue

        # The net learns what the baseline got wrong on the training fold.
        train_resid = train[TARGET_COLUMN] - base_train
        xtr, gtr, ytr, _ = build_matrices(train, train_resid, hawkes_ratio)
        if len(ytr) < cfg.nn.batch_size:
            continue

        if trainer is None:
            net = build_net(len(FEATURE_COLUMNS), len(GATE_COLUMNS), cfg.nn)
            trainer = OnlineTrainer(net, cfg.nn, len(FEATURE_COLUMNS), len(GATE_COLUMNS))
            trainer.warmup(xtr, gtr, ytr)
        else:
            # Realised errors on the fold just trained drive the CUSUM detector.
            prior = trainer.predict(xtr, gtr)
            trainer.observe(xtr, gtr, ytr, errors=ytr - prior)

        xte, gte, _, test_idx = build_matrices(
            test, pd.Series(0.0, index=test.index), hawkes_ratio
        )
        if len(test_idx) == 0:
            continue
        nn_out = pd.Series(trainer.predict(xte, gte), index=test_idx, dtype="float64")
        nn_preds.loc[nn_out.index] = nn_out.to_numpy()

        # Blend using the IC earned on *previous* folds only, then update the record
        # with this fold's outcome. Using this fold's own IC would be lookahead.
        w = blend_state.nn_weight
        blended_preds.loc[test.index] = blend(base_test, nn_out, w).to_numpy()

        realised_resid = (test[TARGET_COLUMN] - base_test).dropna()
        if len(realised_resid) >= 30:
            ic = blend_state.update(nn_out.reindex(realised_resid.index), realised_resid)
            result.ic_history.append((test_hi, ic, w))

    result.regime_shifts = trainer.regime_shifts if trainer else 0
    return result
