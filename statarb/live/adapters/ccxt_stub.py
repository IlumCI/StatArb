"""Live exchange adapter stub. Not wired, and refuses to trade by default.

This exists so the live path has a concrete shape without this repository ever being
one config flag away from sending real orders. Three independent conditions must all
hold before it will even construct:

1. ``allow_live=True`` passed explicitly in code,
2. the ``STATARB_ALLOW_LIVE`` environment variable set to ``1``, and
3. API credentials present in the environment.

Even then :meth:`submit` raises, because no order-routing logic is implemented here.
Wiring it is a deliberate act for the operator, not something that can happen by
accident, and nothing in this repo should be traded with real money on the strength of
the backtest in the report: it does not show a profitable strategy.
"""

from __future__ import annotations

import os

from statarb.live.adapters.base import ExecutionAdapter, Fill, Order

ENV_ALLOW = "STATARB_ALLOW_LIVE"
ENV_KEY = "STATARB_API_KEY"
ENV_SECRET = "STATARB_API_SECRET"


class LiveTradingDisabled(RuntimeError):
    """Raised whenever live trading is attempted without full explicit opt-in."""


class CcxtAdapter(ExecutionAdapter):
    name = "ccxt"
    is_live = True

    def __init__(self, exchange_id: str = "binance", *, allow_live: bool = False):
        if not allow_live:
            raise LiveTradingDisabled(
                "CcxtAdapter requires allow_live=True. Use PaperAdapter for simulation."
            )
        if os.environ.get(ENV_ALLOW) != "1":
            raise LiveTradingDisabled(f"{ENV_ALLOW}=1 is required to construct a live adapter.")
        if not os.environ.get(ENV_KEY) or not os.environ.get(ENV_SECRET):
            raise LiveTradingDisabled(f"{ENV_KEY} and {ENV_SECRET} must be set for live trading.")
        self.exchange_id = exchange_id

    def submit(self, order: Order, reference_price: float, cost_bp: float) -> Fill:
        raise NotImplementedError(
            "Live order routing is intentionally not implemented. Implement submit() "
            "against your venue's API, add position reconciliation and kill-switch "
            "handling, and test on that venue's testnet before considering real funds."
        )
