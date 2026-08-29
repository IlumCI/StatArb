"""Purged walk-forward splitting.

A plain rolling split leaks: the target at the last training bar is a *forward*
return that overlaps bars in the test window, so the model gets to see part of its
own test set. Purging removes those overlapping bars and an embargo drops a further
gap after the boundary, which also blunts autocorrelation carrying across the split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int
    train_end: int   # exclusive, already purged
    test_start: int
    test_end: int    # exclusive

    def describe(self, idx: pd.DatetimeIndex) -> str:
        return (
            f"fold {self.index}: train {idx[self.train_start]:%Y-%m-%d} "
            f"-> {idx[self.train_end - 1]:%Y-%m-%d} ({self.train_end - self.train_start} bars), "
            f"test {idx[self.test_start]:%Y-%m-%d} -> {idx[self.test_end - 1]:%Y-%m-%d} "
            f"({self.test_end - self.test_start} bars)"
        )


def walk_forward_folds(
    n: int, train_bars: int, test_bars: int, embargo_bars: int, holding_bars: int = 1
) -> list[Fold]:
    """Generate expanding-origin, rolling-window folds over ``n`` bars.

    The training window ends ``holding_bars + embargo_bars`` before the test window
    starts. ``holding_bars`` covers the target overlap; ``embargo_bars`` is the extra
    buffer.
    """
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    purge = max(holding_bars, 1) + max(embargo_bars, 0)
    folds: list[Fold] = []
    test_start = train_bars + purge
    i = 0
    while test_start < n:
        test_end = min(test_start + test_bars, n)
        train_end = test_start - purge
        train_start = max(0, train_end - train_bars)
        if train_end - train_start >= max(train_bars // 4, 50):
            folds.append(Fold(i, train_start, train_end, test_start, test_end))
            i += 1
        test_start = test_end
    return folds


def split_long_frame(
    frame: pd.DataFrame, index: pd.DatetimeIndex, fold: Fold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a long-format (timestamp, symbol) frame into train and test parts."""
    ts = frame.index.get_level_values("timestamp")
    train_lo, train_hi = index[fold.train_start], index[fold.train_end - 1]
    test_lo, test_hi = index[fold.test_start], index[fold.test_end - 1]
    train = frame[(ts >= train_lo) & (ts <= train_hi)]
    test = frame[(ts >= test_lo) & (ts <= test_hi)]
    return train, test


def assert_no_overlap(folds: list[Fold], holding_bars: int, embargo_bars: int) -> None:
    """Fail loudly if any fold's training data could touch its test window."""
    required = max(holding_bars, 1) + max(embargo_bars, 0)
    for f in folds:
        gap = f.test_start - f.train_end
        if gap < required:
            raise AssertionError(
                f"fold {f.index}: gap {gap} bars < required purge {required}"
            )
        if f.train_end > f.test_start:
            raise AssertionError(f"fold {f.index}: train window overlaps test window")


def coverage(folds: list[Fold], n: int) -> float:
    """Fraction of the sample that is actually evaluated out of sample."""
    if not folds:
        return 0.0
    covered = np.zeros(n, dtype=bool)
    for f in folds:
        covered[f.test_start : f.test_end] = True
    return float(covered.mean())
