"""Shared design matrix.

The baseline regression, the Hawkes gate and the neural net all consume the same
long-format feature table. Building it in exactly one place is what makes the
no-lookahead guarantee testable: there is a single function to audit, and a single
place where a future-looking column could ever sneak in.

Layout is long format, MultiIndex ``(timestamp, symbol)``, one row per follower per
bar. Every feature column is knowable at the close of ``timestamp``; the target
columns are strictly forward looking and are never fed back in as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config import SignalConfig
from statarb.data.panel import Panel
from statarb.features.leadlag import gate_mask, leader_features
from statarb.features.micro import (
    amihud_illiquidity,
    rolling_dollar_volume,
    staleness_ratio,
)
from statarb.features.returns import (
    cross_sectional_rank,
    forward_return,
    intrabar_returns,
    log_returns,
    realised_vol,
)

#: Columns fed to models. Kept explicit so an accidental extra column cannot leak in.
FEATURE_COLUMNS: tuple[str, ...] = (
    "leader_intrabar",
    "leader_intrabar_z",
    "leader_vol",
    "leader_cum_2",
    "leader_cum_6",
    "own_ret_1",
    "own_ret_6",
    "own_vol",
    "beta_hat",
    "resid_gap",
    "staleness",
    "illiq_z",
    "dollar_vol_log",
    "rank_resid_gap",
)

TARGET_COLUMN = "fwd_return"
LEADER_TARGET_COLUMN = "leader_fwd_return"


@dataclass(frozen=True)
class Dataset:
    """Long-format features and targets plus the wide frames the backtest needs."""

    frame: pd.DataFrame          # MultiIndex (timestamp, symbol)
    gate: pd.Series              # per-timestamp: leader move exceeded trailing quantile
    dollar_volume: pd.DataFrame  # wide, trailing mean dollar volume per bar
    leader_intrabar: pd.Series   # wide, leader open->close return per bar
    index: pd.DatetimeIndex
    followers: tuple[str, ...]

    @property
    def features(self) -> pd.DataFrame:
        return self.frame[list(FEATURE_COLUMNS)]

    @property
    def target(self) -> pd.Series:
        return self.frame[TARGET_COLUMN]

    def wide(self, column: str) -> pd.DataFrame:
        """Pivot one long column back to timestamp x symbol."""
        return self.frame[column].unstack(level="symbol")


def _to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stacked = {
        name: f.stack(future_stack=True) if _supports_future_stack() else f.stack(dropna=False)
        for name, f in frames.items()
    }
    out = pd.DataFrame(stacked)
    out.index = out.index.set_names(["timestamp", "symbol"])
    return out


def _supports_future_stack() -> bool:
    return tuple(int(p) for p in pd.__version__.split(".")[:2]) >= (2, 1)


def build_dataset(panel: Panel, cfg: SignalConfig, *, with_targets: bool = True) -> Dataset:
    """Assemble the full feature table from an aligned panel.

    ``with_targets=False`` is used by the live loop, where the forward return does not
    exist yet; the feature columns are computed identically either way.
    """
    followers = list(panel.followers)
    close = panel.close[followers]
    open_ = panel.open[followers]
    volume = panel.volume[followers]

    lead = leader_features(
        panel.leader_series("open"),
        panel.leader_series("close"),
        cfg.leader_lags,
        cfg.beta_window,
    )
    leader_c2c = np.log(panel.leader_series("close")).diff()

    own_c2c = log_returns(close)
    own_intrabar = intrabar_returns(open_, close)

    # Rolling beta of each follower on the leader, estimated on trailing data only and
    # shifted so bar t uses coefficients fitted strictly before t.
    mp = cfg.beta_min_periods
    cov = own_c2c.rolling(cfg.beta_window, min_periods=mp).cov(leader_c2c)
    var = leader_c2c.rolling(cfg.beta_window, min_periods=mp).var()
    beta = cov.div(var.replace(0.0, np.nan), axis=0).shift(1)

    # The catch-up gap: how much of the leader's move this bar the follower has *not*
    # yet reflected. This is the economic heart of the signal.
    expected = beta.mul(lead["leader_intrabar"], axis=0)
    resid_gap = expected - own_intrabar

    dvol = rolling_dollar_volume(close, volume, cfg.beta_window, min_periods=mp)
    illiq = amihud_illiquidity(own_c2c, close, volume, cfg.beta_window, min_periods=mp)
    stale = staleness_ratio(close, cfg.beta_window, min_periods=mp)
    own_vol = realised_vol(own_c2c, cfg.beta_window, min_periods=mp)

    illiq_z = (illiq.sub(illiq.mean(axis=1), axis=0)).div(illiq.std(axis=1).replace(0.0, np.nan), axis=0)

    wide: dict[str, pd.DataFrame] = {
        "own_ret_1": own_c2c,
        "own_ret_6": own_c2c.rolling(6, min_periods=1).sum(),
        "own_vol": own_vol,
        "beta_hat": beta,
        "resid_gap": resid_gap,
        "staleness": stale,
        "illiq_z": illiq_z,
        "dollar_vol_log": np.log1p(dvol.clip(lower=0.0)),
        "rank_resid_gap": cross_sectional_rank(resid_gap),
        "own_intrabar": own_intrabar,
    }
    # Broadcast the leader's (single-series) features across every follower column.
    for col in ("leader_intrabar", "leader_intrabar_z", "leader_vol", "leader_cum_2", "leader_cum_6"):
        wide[col] = pd.DataFrame(
            np.repeat(lead[col].to_numpy()[:, None], len(followers), axis=1),
            index=close.index, columns=followers,
        )

    if with_targets:
        wide[TARGET_COLUMN] = forward_return(
            open_, close, entry_lag=1, holding_bars=cfg.holding_bars
        )
        leader_fwd = forward_return(
            panel.open[[panel.leader]], panel.close[[panel.leader]],
            entry_lag=1, holding_bars=cfg.holding_bars,
        )[panel.leader]
        wide[LEADER_TARGET_COLUMN] = pd.DataFrame(
            np.repeat(leader_fwd.to_numpy()[:, None], len(followers), axis=1),
            index=close.index, columns=followers,
        )

    frame = _to_long(wide)

    gate = gate_mask(
        lead["leader_intrabar"], cfg.gate_window, cfg.gate_quantile, cfg.gate_min_periods
    )

    return Dataset(
        frame=frame,
        gate=gate.reindex(close.index).fillna(False),
        dollar_volume=dvol,
        leader_intrabar=lead["leader_intrabar"],
        index=pd.DatetimeIndex(close.index),
        followers=tuple(followers),
    )
