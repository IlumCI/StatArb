"""Live loop, adapters and blotter."""

from __future__ import annotations

import pytest

from statarb.live.adapters.base import Order
from statarb.live.adapters.ccxt_stub import CcxtAdapter, LiveTradingDisabled
from statarb.live.adapters.paper import PaperAdapter
from statarb.live.blotter import Blotter


def test_paper_fill_is_pessimistic():
    """A buy must fill above the reference and a sell below it."""
    a = PaperAdapter()
    buy = a.submit(Order("X", "buy", 1000.0), 100.0, 50.0)
    sell = a.submit(Order("X", "sell", 1000.0), 100.0, 50.0)
    assert buy.price > 100.0
    assert sell.price < 100.0
    assert buy.simulated and sell.simulated
    assert buy.quantity == pytest.approx(1000.0 / buy.price)


def test_paper_rejects_invalid_price():
    with pytest.raises(ValueError, match="invalid reference price"):
        PaperAdapter().submit(Order("X", "buy", 100.0), 0.0, 10.0)


def test_live_adapter_refuses_without_explicit_optin(monkeypatch):
    monkeypatch.delenv("STATARB_ALLOW_LIVE", raising=False)
    with pytest.raises(LiveTradingDisabled):
        CcxtAdapter()
    with pytest.raises(LiveTradingDisabled, match="STATARB_ALLOW_LIVE"):
        CcxtAdapter(allow_live=True)


def test_live_adapter_refuses_without_credentials(monkeypatch):
    monkeypatch.setenv("STATARB_ALLOW_LIVE", "1")
    monkeypatch.delenv("STATARB_API_KEY", raising=False)
    monkeypatch.delenv("STATARB_API_SECRET", raising=False)
    with pytest.raises(LiveTradingDisabled, match="STATARB_API_KEY"):
        CcxtAdapter(allow_live=True)


def test_live_adapter_still_refuses_to_route_with_full_optin(monkeypatch):
    """Even fully opted in, order routing is deliberately unimplemented."""
    monkeypatch.setenv("STATARB_ALLOW_LIVE", "1")
    monkeypatch.setenv("STATARB_API_KEY", "k")
    monkeypatch.setenv("STATARB_API_SECRET", "s")
    adapter = CcxtAdapter(allow_live=True)
    with pytest.raises(NotImplementedError, match="not implemented"):
        adapter.submit(Order("X", "buy", 1.0), 1.0, 1.0)


def test_blotter_is_append_only(tmp_path):
    b = Blotter(tmp_path / "b.jsonl")
    b.record("intent", {"symbol": "A", "notional": 1.0})
    b.record("fill", {"symbol": "A", "price": 2.0})
    df = b.read()
    assert len(df) == 2
    assert list(df["kind"]) == ["intent", "fill"]
    assert "logged_at" in df.columns


def test_live_cycle_produces_orders_on_synthetic_data(tmp_path, synthetic_bars, fast_config, monkeypatch):
    """A full cycle against cached synthetic bars must route orders when gated open."""
    import dataclasses

    from statarb.config import LiveConfig
    from statarb.data.cache import write_cache
    from statarb.live.scheduler import LiveTrader
    from statarb.universe import Universe

    cache = tmp_path / "cache"
    for sym, frame in synthetic_bars.items():
        write_cache(cache, sym, "1h", frame)

    cfg = fast_config.with_overrides(
        data=dataclasses.replace(fast_config.data, cache_dir=cache),
        live=LiveConfig(
            blotter_path=tmp_path / "blotter.jsonl",
            state_path=tmp_path / "state.json",
            max_cycles=1,
        ),
    )
    universe = Universe("LEAD-USD", tuple(s for s in synthetic_bars if s != "LEAD-USD"))
    trader = LiveTrader(cfg, universe=universe)
    summary = trader.run_cycle(dry_run=True, refresh=False)
    assert summary["dry_run"] is True
    assert summary["adapter"] == "paper"
    assert Blotter(cfg.live.blotter_path).read().shape[0] >= 1
