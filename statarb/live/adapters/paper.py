"""Paper execution: simulated fills at a reference price plus modelled cost."""

from __future__ import annotations

from statarb.live.adapters.base import ExecutionAdapter, Fill, Order


class PaperAdapter(ExecutionAdapter):
    """Fills every order at the reference price, degraded by the modelled cost.

    Deliberately pessimistic in direction: a buy fills above the reference and a sell
    below it, so paper results cannot look better than the cost model allows.
    """

    name = "paper"
    is_live = False

    def submit(self, order: Order, reference_price: float, cost_bp: float) -> Fill:
        if reference_price <= 0:
            raise ValueError(f"invalid reference price {reference_price} for {order.symbol}")
        slip = cost_bp * 1e-4
        price = reference_price * (1 + slip if order.side == "buy" else 1 - slip)
        return Fill(
            order=order,
            price=price,
            quantity=order.notional / price,
            cost_bp=cost_bp,
            simulated=True,
        )
