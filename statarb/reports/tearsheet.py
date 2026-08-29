"""Report generation.

The headline number is deliberately **breakeven cost**: the round-trip cost at which
the measured gross edge nets to zero. Comparing that against the cost the venue would
actually charge answers the only question that matters, and it does so without
hiding behind an equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.backtest.engine import BacktestResult


def to_markdown_table(frame: pd.DataFrame, *, index_name: str = "") -> str:
    """Render a DataFrame as a markdown table.

    Hand-rolled rather than using ``DataFrame.to_markdown``, which pulls in tabulate;
    the reports are the deliverable and should not need an extra dependency to build.
    """
    if frame.empty:
        return "_(no rows)_"
    header = [index_name or (frame.index.name or "")] + [str(c) for c in frame.columns]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            if not np.isfinite(v):
                return "n/a"
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
        return str(v)

    rows = [[str(idx)] + [fmt(v) for v in frame.loc[idx]] for idx in frame.index]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |"]
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |")
    for r in rows:
        out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(header))) + " |")
    return "\n".join(out)


@dataclass
class ReportSection:
    title: str
    frame: pd.DataFrame
    note: str = ""


def results_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        s = r.stats
        rows[name] = {
            "trades": s.n_trades,
            "gross_bp_per_trade": s.mean_trade_bp,
            "clustered_se_bp": s.mean_trade_bp_se,
            "clustered_t": s.mean_trade_t,
            "cost_bp_round_trip": s.cost_bp_per_trade,
            "net_bp_per_trade": s.mean_trade_bp - s.cost_bp_per_trade,
            "breakeven_cost_bp": s.breakeven_cost_bp,
            "gross_sharpe": s.gross_sharpe,
            "net_sharpe": s.net_sharpe,
            "net_sharpe_lo": s.net_sharpe_ci[0],
            "net_sharpe_hi": s.net_sharpe_ci[1],
            "deflated_sharpe": s.deflated_sharpe,
            "hit_rate": s.hit_rate,
            "max_drawdown": s.max_drawdown,
        }
    return pd.DataFrame(rows).T


def verdict(results: dict[str, BacktestResult]) -> str:
    """One-line assessment. Deliberately blunt, and willing to say no."""
    lines = []
    for name, r in results.items():
        s = r.stats
        net = s.mean_trade_bp - s.cost_bp_per_trade
        significant = np.isfinite(s.mean_trade_t) and abs(s.mean_trade_t) > 2.0
        if net > 0 and significant:
            lines.append(f"{name}: TRADABLE - net {net:+.2f}bp/trade, clustered t={s.mean_trade_t:+.2f}")
        elif s.mean_trade_bp > 0 and significant:
            lines.append(
                f"{name}: NOT TRADABLE - gross edge {s.mean_trade_bp:+.2f}bp (t={s.mean_trade_t:+.2f}) "
                f"is real but costs {s.cost_bp_per_trade:.1f}bp, so it nets {net:+.2f}bp"
            )
        else:
            lines.append(
                f"{name}: NO EDGE - gross {s.mean_trade_bp:+.2f}bp is not distinguishable "
                f"from zero (clustered t={s.mean_trade_t:+.2f})"
            )
    return "\n".join(lines)


def markdown_report(
    results: dict[str, BacktestResult],
    *,
    leadlag: pd.DataFrame | None = None,
    hawkes: pd.DataFrame | None = None,
    capacity: pd.DataFrame | None = None,
    ablation: pd.DataFrame | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    parts = ["# Lead-Lag Statistical Arbitrage - Results\n"]
    parts.append("## Verdict\n\n```\n" + verdict(results) + "\n```\n")
    parts.append("## Performance\n\n" + to_markdown_table(results_table(results).round(4), index_name="construction") + "\n")
    if leadlag is not None:
        parts.append("## Lead-lag structure\n\n" + to_markdown_table(leadlag.round(4), index_name="symbol") + "\n")
    if hawkes is not None:
        parts.append("## Hawkes crash model\n\n" + to_markdown_table(hawkes.round(4), index_name="symbol") + "\n")
    if capacity is not None:
        parts.append("## Capacity curve\n\n" + to_markdown_table(capacity.round(3)) + "\n")
    if ablation is not None:
        parts.append("## Component ablation\n\n" + to_markdown_table(ablation.round(3), index_name="variant") + "\n")
    for title, body in (extra or {}).items():
        parts.append(f"## {title}\n\n{body}\n")
    return "\n".join(parts)


def equity_plot(results: dict[str, BacktestResult], path: str) -> str | None:
    """Cumulative net return per construction. Returns None if matplotlib is absent."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:  # pragma: no cover - optional dependency
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        ax.plot(r.equity_curve.index, r.equity_curve.to_numpy(), label=f"{name} (net)")
        ax.plot(
            r.bar_returns_gross.fillna(0).cumsum().index,
            r.bar_returns_gross.fillna(0).cumsum().to_numpy(),
            linestyle="--", alpha=0.6, label=f"{name} (gross)",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative return, gross vs net of costs")
    ax.set_ylabel("cumulative log return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
