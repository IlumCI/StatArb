"""Command line interface.

    statarb fetch                     populate the bar cache
    statarb research leadlag          cross-correlation table
    statarb research hawkes           fitted branching ratios
    statarb backtest                  walk-forward backtest and report
    statarb capacity                  net edge as a function of book size
    statarb paper                     live paper-trading loop
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from statarb.config import DEFAULT_ARTIFACT_DIR, Config
from statarb.data.cache import load_universe
from statarb.data.panel import build_panel
from statarb.universe import average_dollar_volume, get_market, static_tiers

log = logging.getLogger("statarb")


def _load_panel(cfg: Config, *, refresh: bool):
    spec = get_market(cfg.market)
    bars = load_universe(
        spec.symbols, interval=cfg.data.interval, lookback=cfg.data.lookback,
        cache_dir=cfg.data.cache_dir, backend=cfg.data.backend, refresh=refresh,
    )
    return build_panel(
        bars, leader=spec.leader, followers=spec.followers, interval=cfg.data.interval,
        min_cross_section_coverage=cfg.data.min_cross_section_coverage,
        session_tz=spec.session_tz,
    )


def _config(args) -> Config:
    if getattr(args, "config", None):
        cfg = Config.load(args.config)
    else:
        cfg = Config.for_market(getattr(args, "market", None) or "solana")
    port = {}
    if getattr(args, "notional", None):
        port["gross_notional"] = args.notional
    if getattr(args, "construction", None):
        port["construction"] = args.construction
    if port:
        cfg = cfg.with_overrides(portfolio=dataclasses.replace(cfg.portfolio, **port))
    if getattr(args, "no_nn", False):
        cfg = cfg.with_overrides(nn=dataclasses.replace(cfg.nn, enabled=False))
    return cfg


def cmd_fetch(args) -> int:
    cfg = _config(args)
    panel = _load_panel(cfg, refresh=not args.no_refresh)
    adv = average_dollar_volume(panel.close, panel.volume, cfg.signal.beta_window)
    desc = panel.describe()
    desc["tier"] = static_tiers(adv)
    print(f"market: {cfg.market} | panel: {panel.close.shape[0]} bars x {panel.close.shape[1]} symbols")
    print(f"range: {panel.index[0]} -> {panel.index[-1]}")
    if panel.session_tz:
        from statarb.data.sessions import session_summary  # noqa: PLC0415

        print(f"sessions: {session_summary(panel.sessions)}")
    print(desc.to_string())
    return 0


def cmd_research(args) -> int:
    cfg = _config(args)
    panel = _load_panel(cfg, refresh=not args.no_refresh)
    from statarb.data.sessions import mask_cross_session  # noqa: PLC0415
    from statarb.features.returns import log_returns  # noqa: PLC0415

    # Mask returns spanning a session boundary, exactly as the model does. Measuring
    # the diagnostic on raw close-to-close would fold in overnight gaps and report a
    # different, and in equities oppositely signed, coefficient from the one traded.
    lr = mask_cross_session(log_returns(panel.close), panel.sessions)

    if args.topic == "leadlag":
        from statarb.features.leadlag import cross_correlation  # noqa: PLC0415

        table = cross_correlation(lr[panel.leader], lr[list(panel.followers)], max_lag=3)
        adv = average_dollar_volume(panel.close, panel.volume, cfg.signal.beta_window).mean()
        table["adv_per_bar"] = adv
        table = table.sort_values("adv_per_bar", ascending=False)
        print(table.round(4).to_string())
        n_sig = int((table["t_lag1"] > 2).sum())
        half = max(len(table) // 2, 1)
        liquid = table["lag1"].iloc[:half].mean()
        illiquid = table["lag1"].iloc[half:].mean()
        print(
            f"\n{n_sig}/{len(table)} followers significant at t > 2 "
            f"| liquid half mean lag1 {liquid:+.4f} vs illiquid half {illiquid:+.4f}"
        )
        print(
            "A lead-lag that strengthens as liquidity falls, with reverse causality near "
            "zero, is the signature of stale pricing rather than simple co-movement. "
            "A flat gradient means the signal is not concentrated in unfillable names."
        )
    elif args.topic == "hawkes":
        from statarb.models.hawkes import fit_bivariate_hawkes, fit_hawkes, jump_events
        from statarb.risk import trailing_jump_threshold

        thresholds = trailing_jump_threshold(lr, cfg.hawkes)
        lead_thr = trailing_jump_threshold(lr[[panel.leader]], cfg.hawkes)[panel.leader]
        lead_ev = jump_events(lr[panel.leader], lead_thr)
        horizon = float(len(panel.index))
        rows = {}
        for sym in panel.followers:
            ev = jump_events(lr[sym], thresholds[sym])
            fp = fit_hawkes(
                ev, horizon, mu_bounds=cfg.hawkes.mu_bounds,
                alpha_bounds=cfg.hawkes.alpha_bounds, beta_bounds=cfg.hawkes.beta_bounds,
            )
            bp = fit_bivariate_hawkes(ev, lead_ev, horizon)
            rows[sym] = {
                "events": fp.n_events, "mu": fp.mu, "branching_n": fp.branching_ratio,
                "cascade_size": fp.expected_cascade_size, "half_life_bars": fp.half_life,
                "cross_n_from_leader": bp.cross_branching,
            }
        print(pd.DataFrame(rows).T.sort_values("branching_n", ascending=False).round(4).to_string())
        print(
            "\nbranching_n is the expected number of aftershocks each crash triggers; "
            "as it approaches 1 the asset is one shock from a cascade."
        )
    return 0


def cmd_backtest(args) -> int:
    cfg = _config(args)
    panel = _load_panel(cfg, refresh=not args.no_refresh)
    from statarb.backtest.engine import CONSTRUCTORS, prepare, run_construction
    from statarb.models.nn.runner import run_overlay
    from statarb.reports.tearsheet import equity_plot, markdown_report, results_table, verdict

    prep = prepare(panel, cfg)
    predictions = prep.predictions
    overlay_note = "neural overlay disabled"
    if cfg.nn.enabled:
        ratio = prep.hawkes.intensity_ratio.stack(future_stack=True)
        ratio.index = ratio.index.set_names(["timestamp", "symbol"])
        overlay = run_overlay(prep.dataset, cfg, ratio)
        if overlay.trained:
            predictions = overlay.blended_predictions
            ic = overlay.ic_frame()
            overlay_note = (
                f"mean out-of-sample IC {ic['ic'].mean():+.4f} over {len(ic)} folds, "
                f"{(ic['ic'] > 0).sum()}/{len(ic)} positive; final blend weight "
                f"{ic['nn_weight'].iloc[-1]:.2f}; {overlay.regime_shifts} regime shifts detected"
            )
    prep = dataclasses.replace(prep, predictions=predictions)

    which = list(CONSTRUCTORS) if cfg.portfolio.construction == "both" else [cfg.portfolio.construction]
    results = {c: run_construction(prep, cfg, c) for c in which}

    print(results_table(results).round(4).to_string())
    print("\n" + verdict(results))

    out = Path(args.out or DEFAULT_ARTIFACT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    report = markdown_report(results, extra={"Neural overlay": overlay_note})
    (out / "report.md").write_text(report)
    equity_plot(results, str(out / "equity.png"))
    (out / "stats.json").write_text(
        json.dumps({k: v.stats.to_dict() for k, v in results.items()}, indent=2, default=str)
    )
    print(f"\nwrote {out/'report.md'}, {out/'stats.json'}, {out/'equity.png'}")
    return 0


def cmd_capacity(args) -> int:
    """Net edge as a function of book size: where, if anywhere, this is tradable."""
    cfg = _config(args)
    panel = _load_panel(cfg, refresh=not args.no_refresh)
    from statarb.backtest.engine import prepare, run_construction

    prep = prepare(panel, cfg)
    rows = []
    for notional in [1e3, 1e4, 1e5, 1e6, 1e7]:
        c = cfg.with_overrides(
            portfolio=dataclasses.replace(cfg.portfolio, gross_notional=notional)
        )
        for con in ("cross_sectional", "directional"):
            s = run_construction(prep, c, con).stats
            rows.append(
                {
                    "notional": notional, "construction": con, "trades": s.n_trades,
                    "gross_bp": s.mean_trade_bp, "clustered_t": s.mean_trade_t,
                    "cost_bp": s.cost_bp_per_trade,
                    "net_bp": s.mean_trade_bp - s.cost_bp_per_trade,
                    "net_sharpe": s.net_sharpe,
                }
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


def cmd_paper(args) -> int:
    cfg = _config(args)
    if args.cycles:
        cfg = cfg.with_overrides(live=dataclasses.replace(cfg.live, max_cycles=args.cycles))
    from statarb.live.scheduler import LiveTrader

    trader = LiveTrader(cfg)
    if args.once:
        print(json.dumps(trader.run_cycle(dry_run=args.dry_run, refresh=not args.no_refresh), indent=2))
    else:
        for summary in trader.run(dry_run=args.dry_run):
            print(json.dumps(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="statarb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to a saved JSON config")
    p.add_argument("--market", choices=["solana", "equity"], default="solana",
                   help="which universe to trade (default: solana)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-refresh", action="store_true", help="use cached bars only, no network")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="populate the bar cache and show coverage")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("research", help="diagnostics on the lead-lag and crash structure")
    r.add_argument("topic", choices=["leadlag", "hawkes"])
    r.set_defaults(func=cmd_research)

    b = sub.add_parser("backtest", help="walk-forward backtest and report")
    b.add_argument("--construction", choices=["cross_sectional", "directional", "both"])
    b.add_argument("--notional", type=float, help="gross book size in USD")
    b.add_argument("--no-nn", action="store_true", help="skip the neural overlay")
    b.add_argument("--out", help="artifact directory")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("capacity", help="net edge as a function of book size")
    c.add_argument("--construction")
    c.set_defaults(func=cmd_capacity)

    pa = sub.add_parser("paper", help="paper-trading loop")
    pa.add_argument("--once", action="store_true", help="run a single cycle and exit")
    pa.add_argument("--dry-run", action="store_true", help="log intents without filling")
    pa.add_argument("--cycles", type=int, help="stop after this many cycles")
    pa.add_argument("--construction")
    pa.add_argument("--notional", type=float)
    pa.set_defaults(func=cmd_paper)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    if not args.verbose:
        warnings.filterwarnings("ignore")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
