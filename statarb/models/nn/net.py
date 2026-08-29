"""Regime-gated mixture-of-experts network.

The "self-adapting" part lives in two places. Architecturally, a gating network reads
the current regime (leader volatility, Hawkes intensity, liquidity) and mixes a set of
expert heads, so the effective function changes with market state rather than being
one fixed mapping. Temporally, :mod:`statarb.models.nn.online` keeps updating the
weights as new bars arrive.

The net predicts the *residual* left over by the linear baseline, never the raw
return. That framing matters: it cannot win by rediscovering what the baseline
already knows, so any measured improvement is genuinely incremental.

torch is imported lazily so the rest of the system runs without a 527MB dependency.
"""

from __future__ import annotations

from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def build_net(n_features: int, n_gate_features: int, cfg: Any):
    """Construct the gated MoE. Imported lazily to keep torch optional."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class GatedMixtureOfExperts(nn.Module):
        """Shared trunk, K expert heads, softmax gate over regime features."""

        def __init__(self) -> None:
            super().__init__()
            h, k = cfg.hidden, cfg.n_experts
            self.trunk = nn.Sequential(
                nn.Linear(n_features, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(h, h),
                nn.LayerNorm(h),
                nn.GELU(),
            )
            # Each expert is a linear read-out of the shared representation; capacity
            # lives in the mixture, not in deep per-expert stacks, which keeps the
            # parameter count sane for ~250k rows of noisy financial data.
            self.experts = nn.Linear(h, k)
            self.gate = nn.Sequential(
                nn.Linear(n_gate_features, max(8, k * 2)),
                nn.GELU(),
                nn.Linear(max(8, k * 2), k),
            )
            self.scale = nn.Parameter(torch.tensor(1e-3))

        def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            z = self.trunk(x)
            per_expert = self.experts(z)
            weights = torch.softmax(self.gate(g), dim=-1)
            mixed = (per_expert * weights).sum(dim=-1)
            # Outputs are on the scale of hourly residual returns (~1e-3), so an
            # explicit learned scale keeps the net from having to reach tiny values
            # through the weights alone.
            return mixed * self.scale

        def gate_weights(self, g: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.gate(g), dim=-1)

    return GatedMixtureOfExperts()
