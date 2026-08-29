"""Configuration objects. Single source of truth for every tunable in the system.

Everything that could bias a backtest lives here explicitly rather than being
buried as a magic number, so that a reader can audit the assumptions in one place.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class DataConfig:
    """Bar sourcing.

    Yahoo caps hourly history at ~730 days, which is what makes 1h the sweet spot:
    ~17.5k bars per asset, enough for Hawkes MLE and for a small net, while still
    being fine enough that lead-lag has not been arbitraged away.
    """

    interval: str = "1h"
    lookback: str = "730d"
    cache_dir: Path = DEFAULT_CACHE_DIR
    backend: str = "chart"  # "chart" (direct v8 API) | "yfinance"
    max_retries: int = 4
    retry_backoff_s: float = 2.0
    request_timeout_s: float = 40.0
    # Drop any bar timestamp where fewer than this fraction of the universe traded.
    min_cross_section_coverage: float = 0.6


@dataclass(frozen=True)
class SignalConfig:
    """Lead-lag signal construction."""

    # Leader move horizons (in bars) fed to the baseline regression.
    leader_lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    # Rolling window (bars) for beta / lead-lag coefficient estimation.
    beta_window: int = 720  # ~30 days of hourly bars
    beta_min_periods: int = 240
    # Trailing window for the leader-move quantile gate. Trailing, never full-sample.
    gate_window: int = 2160  # ~90 days
    gate_quantile: float = 0.90
    gate_min_periods: int = 720
    # Holding period in bars. Empirically the edge peaks at t+2 and decays by t+3.
    holding_bars: int = 2
    # Winsorise feature and target returns at this trailing z to blunt data errors.
    winsor_z: float = 6.0


@dataclass(frozen=True)
class HawkesConfig:
    """Self-exciting crash filter."""

    # Jump threshold: trailing quantile of |return|. Trailing, never full-sample.
    jump_quantile: float = 0.95
    jump_window: int = 2160
    jump_min_periods: int = 720
    # Refit cadence (bars) and the trailing sample used for each fit.
    refit_every: int = 720
    fit_window: int = 4320  # ~180 days
    # Hard gate: skip an asset whose conditional intensity ratio lambda/mu exceeds this.
    max_intensity_ratio: float = 2.5
    # Hard gate: skip an asset whose branching ratio (alpha/beta) exceeds this.
    max_branching_ratio: float = 0.85
    # Soft scaler strength: size multiplier is 1 / (1 + kappa * excess intensity).
    soft_kappa: float = 1.0
    # Optimiser bounds for (mu, alpha, beta) in events-per-bar units.
    mu_bounds: tuple[float, float] = (1e-6, 10.0)
    alpha_bounds: tuple[float, float] = (1e-6, 50.0)
    beta_bounds: tuple[float, float] = (1e-4, 50.0)


@dataclass(frozen=True)
class CostConfig:
    """Cost model.

    Corwin-Schultz collapses to ~0bp on thin crypto OHLC (verified empirically on this
    universe), so it is reported as a diagnostic but never used for charging. Costs are
    charged from an ADV-tiered effective-spread floor plus an Amihud impact term.
    """

    # Half-spread in bp by liquidity tier, tier 0 = most liquid.
    tier_half_spread_bp: tuple[float, ...] = (5.0, 12.5, 25.0, 40.0)
    # Exchange taker fee per side, in bp.
    taker_fee_bp: float = 8.0
    # Amihud impact coefficient: extra bp per unit of (order notional / bar dollar volume).
    impact_coef_bp: float = 25.0
    # Cap modelled impact so a single thin bar cannot dominate the backtest.
    max_impact_bp: float = 150.0


@dataclass(frozen=True)
class PortfolioConfig:
    """Position sizing and risk limits."""

    construction: str = "both"  # "cross_sectional" | "directional" | "both"
    gross_notional: float = 100_000.0
    max_weight: float = 0.25
    # Cross-sectional: fraction of the ranked basket taken on each side.
    quantile: float = 0.30
    min_names_per_side: int = 2
    # Annualised volatility target for the portfolio; None disables vol targeting.
    vol_target_annual: float | None = 0.20
    vol_window: int = 720
    max_leverage: float = 3.0
    bars_per_year: int = 24 * 365
    # Refuse to hold a position larger than this fraction of the asset's average bar
    # dollar volume. Without it the book "trades" names whose hourly volume is a few
    # dollars, at 10,000x participation, which no venue could ever fill.
    max_participation: float = 0.10


@dataclass(frozen=True)
class NNConfig:
    """Self-adapting overlay."""

    enabled: bool = True
    hidden: int = 48
    n_experts: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    warmup_epochs: int = 12
    online_epochs: int = 2
    # Replay buffer: recent bars are oversampled, a reservoir keeps older regimes alive.
    buffer_size: int = 40_000
    recent_fraction: float = 0.5
    # CUSUM regime detection scales the learning rate when errors drift.
    cusum_threshold: float = 5.0
    lr_boost: float = 3.0
    # Blend weight on the linear baseline is floored here; the rest is earned by
    # out-of-sample IC, so a useless net collapses to the baseline on its own.
    min_baseline_weight: float = 0.5
    ic_window: int = 2000
    seed: int = 7


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward evaluation."""

    train_bars: int = 4320   # ~180 days
    test_bars: int = 720     # ~30 days
    embargo_bars: int = 24   # purge around the boundary to stop horizon leakage
    block_bootstrap_size: int = 168  # ~1 week blocks
    n_bootstrap: int = 2000
    # Number of distinct configurations tried, used to deflate the Sharpe ratio.
    n_trials_for_deflation: int = 8
    seed: int = 11


@dataclass(frozen=True)
class LiveConfig:
    """Paper / live loop."""

    poll_interval_s: int = 300
    # Wait this long after the bar closes before acting, so the bar is settled.
    bar_settle_lag_s: int = 120
    blotter_path: Path = DEFAULT_ARTIFACT_DIR / "blotter.jsonl"
    state_path: Path = DEFAULT_ARTIFACT_DIR / "live_state.json"
    adapter: str = "paper"  # "paper" | "ccxt"
    max_cycles: int | None = None


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    hawkes: HawkesConfig = field(default_factory=HawkesConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def to_dict(self) -> dict[str, Any]:
        def encode(o: Any) -> Any:
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o

        return json.loads(json.dumps(asdict(self), default=encode))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = json.loads(Path(path).read_text())
        return cls(
            data=DataConfig(**_paths(raw.get("data", {}), {"cache_dir"})),
            signal=SignalConfig(**_tuples(raw.get("signal", {}), {"leader_lags"})),
            hawkes=HawkesConfig(**_tuples(raw.get("hawkes", {}), {"mu_bounds", "alpha_bounds", "beta_bounds"})),
            costs=CostConfig(**_tuples(raw.get("costs", {}), {"tier_half_spread_bp"})),
            portfolio=PortfolioConfig(**raw.get("portfolio", {})),
            nn=NNConfig(**raw.get("nn", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            live=LiveConfig(**_paths(raw.get("live", {}), {"blotter_path", "state_path"})),
        )

    def with_overrides(self, **sections: Any) -> Config:
        """Return a copy with whole sections replaced, e.g. cfg.with_overrides(nn=...)."""
        return replace(self, **sections)


def _paths(d: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: (Path(v) if k in keys and v is not None else v) for k, v in d.items()}


def _tuples(d: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: (tuple(v) if k in keys and v is not None else v) for k, v in d.items()}
