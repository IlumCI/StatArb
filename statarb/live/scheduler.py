"""Live paper-trading loop.

Pulls fresh bars, rebuilds features, produces a target book and routes the difference
to an execution adapter. The model is fit on cached history exactly as in the
backtest, and the same ``t+1 open`` discipline applies: signals computed from the last
*complete* bar are acted on in the next one.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from statarb.backtest.engine import CONSTRUCTORS
from statarb.config import Config
from statarb.costs import cost_breakdown
from statarb.data.cache import load_universe
from statarb.data.panel import build_panel
from statarb.features.dataset import build_dataset
from statarb.features.returns import log_returns
from statarb.live.adapters.base import ExecutionAdapter, Order
from statarb.live.adapters.paper import PaperAdapter
from statarb.live.blotter import Blotter
from statarb.models.baseline import LeadLagBaseline
from statarb.portfolio.base import PortfolioConstructor
from statarb.risk import apply_capacity, capacity_weight_cap, combine_masks, rolling_hawkes_state
from statarb.universe import Universe, get_market

log = logging.getLogger(__name__)


@dataclass
class LiveState:
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> signed notional
    cycles: int = 0
    last_bar: str | None = None

    def to_dict(self) -> dict:
        return {"positions": self.positions, "cycles": self.cycles, "last_bar": self.last_bar}

    @classmethod
    def load(cls, path: Path) -> LiveState:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(raw.get("positions", {}), int(raw.get("cycles", 0)), raw.get("last_bar"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class LiveTrader:
    """One cycle = fetch, signal, size, route. Safe to run repeatedly."""

    def __init__(
        self,
        cfg: Config,
        universe: Universe | None = None,
        adapter: ExecutionAdapter | None = None,
    ):
        self.cfg = cfg
        spec = get_market(cfg.market)
        self.universe = universe or spec.universe
        self.session_tz = spec.session_tz
        self.adapter = adapter or PaperAdapter()
        self.blotter = Blotter(cfg.live.blotter_path)
        self.state = LiveState.load(Path(cfg.live.state_path))
        if self.adapter.is_live:
            log.warning("LIVE adapter in use: orders may reach a real venue")

    def _target_book(self, panel) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
        """Target signed notional per symbol from the most recent complete bar."""
        cfg = self.cfg
        ds = build_dataset(panel, cfg.signal)
        followers = list(panel.followers)

        # Train on everything except the trailing holding window, whose targets have
        # not resolved yet and would otherwise be NaN or partially observed.
        frame = ds.frame
        ts = frame.index.get_level_values("timestamp")
        cutoff = ds.index[-(cfg.signal.holding_bars + 1)]
        train = frame[ts <= cutoff].dropna(subset=["fwd_return"])
        model = LeadLagBaseline().fit(train)

        last_bar = ds.index[-1]
        latest = frame[ts == last_bar]
        scores = model.predict(latest).unstack(level="symbol").reindex(columns=followers)

        hawkes = rolling_hawkes_state(log_returns(panel.close[followers]), cfg.hawkes)
        gate_row = bool(ds.gate.iloc[-1])
        tradable = pd.DataFrame(gate_row, index=scores.index, columns=followers) & scores.notna()
        tradable = combine_masks(
            tradable, hawkes.tradable(cfg.hawkes).reindex(scores.index).reindex(columns=followers)
        )

        ctor: PortfolioConstructor = CONSTRUCTORS[
            cfg.portfolio.construction if cfg.portfolio.construction in CONSTRUCTORS else "cross_sectional"
        ](cfg.portfolio)
        weights = ctor.weights(scores, tradable)
        weights = weights * hawkes.size_scalar(cfg.hawkes).reindex(scores.index).reindex(columns=followers).fillna(0.0)
        cap = capacity_weight_cap(
            ds.dollar_volume.reindex(scores.index).reindex(columns=followers),
            cfg.portfolio.gross_notional,
            cfg.portfolio.max_participation,
        )
        weights = apply_capacity(weights, cap)

        target = weights.iloc[-1] * cfg.portfolio.gross_notional
        dvol = ds.dollar_volume.reindex(columns=followers).iloc[-1]
        return target, dvol, last_bar

    def run_cycle(self, *, dry_run: bool = False, refresh: bool = True) -> dict:
        cfg = self.cfg
        bars = load_universe(
            self.universe.symbols, interval=cfg.data.interval, lookback=cfg.data.lookback,
            cache_dir=cfg.data.cache_dir, backend=cfg.data.backend, refresh=refresh,
        )
        panel = build_panel(
            bars, leader=self.universe.leader, followers=self.universe.followers,
            interval=cfg.data.interval,
            min_cross_section_coverage=cfg.data.min_cross_section_coverage,
            session_tz=self.session_tz,
        )
        target, dvol, last_bar = self._target_book(panel)
        prices = panel.close.iloc[-1]

        # Cost per side for the deltas we are about to trade.
        deltas = {s: float(target.get(s, 0.0) - self.state.positions.get(s, 0.0)) for s in target.index}
        delta_frame = pd.DataFrame([{k: abs(v) for k, v in deltas.items()}])
        bd = cost_breakdown(
            delta_frame, pd.DataFrame([dvol.to_dict()]), cfg.costs
        )

        orders, fills = [], []
        for sym, delta in deltas.items():
            # Skip dust: a ticket below 0.5% of gross is not worth its own spread.
            if abs(delta) < cfg.portfolio.gross_notional * 0.005:
                continue
            price = float(prices.get(sym, np.nan))
            if not np.isfinite(price) or price <= 0:
                log.warning("%s: no valid price, skipping", sym)
                continue
            order = Order(
                symbol=sym, side="buy" if delta > 0 else "sell", notional=abs(delta),
                reason=f"rebalance to {target[sym]:+.2f} at bar {last_bar:%Y-%m-%d %H:%M}",
            )
            orders.append(order)
            cost_bp = float(bd.total_bp.iloc[0].get(sym, cfg.costs.taker_fee_bp))
            if dry_run:
                self.blotter.record("intent", {**order.to_dict(), "cost_bp": cost_bp, "dry_run": True})
                continue
            fill = self.adapter.submit(order, price, cost_bp)
            fills.append(fill)
            self.blotter.record("fill", fill.to_dict())
            self.state.positions[sym] = float(target[sym])

        if not dry_run:
            self.state.cycles += 1
            self.state.last_bar = str(last_bar)
            self.state.save(Path(cfg.live.state_path))

        summary = {
            "bar": str(last_bar),
            "gate_open": bool(target.abs().sum() > 0),
            "n_orders": len(orders),
            "n_fills": len(fills),
            "gross_target": float(target.abs().sum()),
            "dry_run": dry_run,
            "adapter": self.adapter.name,
        }
        self.blotter.record("cycle", summary)
        return summary

    def run(self, *, dry_run: bool = False) -> list[dict]:
        """Loop until ``max_cycles`` is reached, or forever if it is None."""
        out = []
        cfg = self.cfg
        while cfg.live.max_cycles is None or len(out) < cfg.live.max_cycles:
            try:
                out.append(self.run_cycle(dry_run=dry_run))
            except Exception as exc:  # noqa: BLE001 - a live loop must survive one bad cycle
                log.exception("cycle failed: %s", exc)
                self.blotter.record("error", {"error": str(exc)})
            if cfg.live.max_cycles is not None and len(out) >= cfg.live.max_cycles:
                break
            time.sleep(cfg.live.poll_interval_s)
        return out
