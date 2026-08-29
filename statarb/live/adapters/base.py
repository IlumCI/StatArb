"""Execution adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str          # "buy" | "sell"
    notional: float    # always positive, in quote currency
    reason: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "notional": round(self.notional, 6),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Fill:
    order: Order
    price: float
    quantity: float
    cost_bp: float
    simulated: bool

    def to_dict(self) -> dict:
        return {
            **self.order.to_dict(),
            "price": self.price,
            "quantity": self.quantity,
            "cost_bp": round(self.cost_bp, 3),
            "simulated": self.simulated,
        }


class ExecutionAdapter(ABC):
    """Anything that can turn an Order into a Fill."""

    name: str = "base"
    is_live: bool = False

    @abstractmethod
    def submit(self, order: Order, reference_price: float, cost_bp: float) -> Fill: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        return None
