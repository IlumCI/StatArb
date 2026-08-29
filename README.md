# StatArb — Lead-Lag Statistical Arbitrage

A research-grade statistical arbitrage system built on the **lead-lag hypothesis**: a
liquid leader impounds information first, and illiquid followers reprice with a delay.
The system ranks followers by predicted catch-up, filters out assets caught in
self-exciting crash cascades using a **Hawkes process**, and sizes positions under an
explicit capacity constraint.

Two markets are implemented and were tested independently:

- **`solana`** — SOL-USD leading 14 ecosystem tokens, traded continuously. 17,477 hourly bars.
- **`equity`** — IWM leading 27 US small and micro caps, 7 bars per session. 5,072 bars over 730 sessions.

## The headline result

**The lead-lag effect is real in both markets, and tradable in neither — for the same
underlying reason.**

The follower's catch-up completes in the jump between the signal bar's close and the
next bar's open. That is precisely the moment you cannot trade: by the time you can
buy at the next open, the move has already happened.

Correlation of the leader's move at bar `t` with the follower's next-bar move,
decomposed into the untradeable gap and the leg a strategy can actually capture:

| | Equity | Solana |
|---|---|---|
| Total close-to-close predictability | +0.0510 | +0.0361 |
| …realised in the **gap** close(t)→open(t+1) | **+0.0600** | **+0.0787** |
| …left in the **tradable** leg open(t+1)→close(t+1) | +0.0218 | +0.0208 |

The residual tradable correlation of ~0.022 is worth roughly 2bp per trade in both
markets. Costs are 25–47bp. The strategy loses by an order of magnitude.

Each market then fails its own way, which is what makes the pair informative:

| | Solana | Equity |
|---|---|---|
| Gross edge, ignoring capacity | **+8.50 bp** (*t* = 4.22) | +2.15 bp (*t* = 0.83) |
| Gross edge, respecting capacity | **−0.14 bp** (*t* = −0.04) | +1.39 bp (*t* = 0.49) |
| Round-trip cost | ~47 bp | ~25 bp |
| Binding constraint | **capacity** | **weak tradable residual** |

In Solana the apparent edge was large but sat entirely in tokens that cannot be
filled — a $12.5k position in MOBILE-USD is **74,000×** its average hourly volume. In
equities capacity barely binds and costs are half as large, but there was never much
tradable edge to begin with.

That is an economically coherent result rather than a disappointing one: the
inefficiency persists *because* it completes where nobody can trade it. See
[`docs/report.html`](docs/report.html) for the full write-up.

## What is here

```
statarb/
  config.py           every tunable, in one auditable place
  universe.py         leader + follower basket, ADV liquidity tiering
  data/               Yahoo chart client, parquet cache, aligned panel
  features/           returns, microstructure, lead-lag, shared design matrix
  models/
    baseline.py       per-asset ridge on causal features (the benchmark)
    hawkes.py         exponential-kernel MLE, univariate + bivariate
    nn/               regime-gated mixture of experts, online adaptation, IC blending
  risk.py             Hawkes gate, soft size scaler, capacity limits
  costs.py            ADV-tiered spread + Amihud impact + fees
  portfolio/          cross-sectional L/S and directional beta-hedged
  backtest/           purged walk-forward, clustered SEs, block bootstrap
  live/               paper loop, append-only blotter, unwired live stub
  reports/            markdown tearsheet + equity plot
```

## Install and run

```bash
pip install -e ".[dev,report]"      # add ",nn" for the neural overlay (needs torch)

# --market selects the universe (solana | equity) and its venue conventions
statarb --market equity fetch                  # populate the bar cache
statarb --market equity research leadlag       # cross-correlation by liquidity tier
statarb --market equity research hawkes        # fitted branching ratios per asset
statarb --market equity backtest --construction both --notional 1000000
statarb --market equity capacity               # net edge vs book size
statarb --market equity paper --once --dry-run # one live cycle, logs intents only
```

`--market` carries more than a ticker list: session timezone, bars per year, window
sizes expressed in trading time, and a venue-appropriate cost model. Copying the
crypto config onto equities would silently turn a "30 day" beta window into five
months and charge an 8bp taker fee on a market that charges 0.5bp.

## Methodology

The system is built to **avoid flattering itself**. Specifically:

**Execution.** A signal formed at the close of bar `t` executes at the **open of bar
`t+1`**. Filling at the signal bar's own close is lookahead, and on stale illiquid
prices it is the easiest way to manufacture a Sharpe ratio that cannot be traded.
`forward_return` raises if asked for `entry_lag=0`.

**No lookahead, structurally.** Every rolling estimate — beta, volatility, jump
thresholds, Hawkes parameters, liquidity tiers — is trailing and shifted. This is
enforced by a test that corrupts all data after a cut point and asserts that no
feature before it changes.

**Honest inference.** Fourteen followers reacting to the same leader move in one hour
is *one bet, not fourteen*, so per-trade standard errors are **clustered by
timestamp**. Sharpe confidence intervals come from a stationary block bootstrap, and
the deflated Sharpe ratio corrects for the number of configurations tried.

**Purged walk-forward.** 19 folds, 75% out-of-sample coverage, with the training
window ending `holding_bars + embargo_bars` before each test window so overlapping
forward returns cannot leak across the boundary.

**Capacity.** Positions are capped at a configurable fraction of each asset's average
bar dollar volume. This is the constraint that separates a paper edge from a real one,
and here it is what kills the strategy.

**Costs.** ADV-tiered effective spreads, Amihud impact scaled by participation, and
taker fees, all per market. Corwin-Schultz is computed but deliberately *not* used for
charging: it estimates near-zero spreads on thin crypto OHLC, which would flatter the
backtest exactly where reality is worst.

**Sessions.** Equities do not trade continuously, so returns spanning a session
boundary are overnight gaps rather than intraday moves and are nulled out. Positions
are only opened when signal, entry and exit all land in the same session, so the
strategy never silently collects overnight gap risk it is not being paid for. A
continuous market (`session_tz=None`) is one unbroken session and none of this
applies.

## Components

**Hawkes crash filter.** Fits `λ(t) = μ + Σ α·exp(−β(t−tᵢ))` on negative jump arrivals
by maximum likelihood, reporting the branching ratio `α/β` (expected aftershocks per
jump). A bivariate variant adds leader→follower cross-excitation — lead-lag in jump
space. Used as a hard gate and a soft size scaler. Parameter recovery is unit-tested
against simulated processes.

**Self-adapting neural overlay.** A regime-gated mixture of experts predicts the
*residual* the linear baseline misses, so it cannot win by rediscovering what the
baseline already knows. It adapts online via a replay buffer that mixes recent bars
with a reservoir sample of history (preventing catastrophic forgetting), with a CUSUM
detector on prediction error that raises the learning rate on regime shifts. The blend
weight is earned from realised out-of-sample IC, so a net that adds nothing collapses
to the baseline automatically.

## Findings worth recording

1. **The predictable move happens in the untradeable gap.** In both markets the
   close-to-close lead-lag coefficient is roughly double what survives into the
   open-to-close leg a strategy can capture. This is the central result, and it is
   market independent.

2. **In Solana, lead-lag is ordered by illiquidity.** Lag-1 correlation rises
   monotonically as liquidity falls (SAMO 0.108 *t*=14.4 → JTO 0.010 *t*=1.4) while
   lag-0 falls in step, and reverse causality is ≈0 everywhere.

3. **In equities the gradient is flat, so capacity is not the binding constraint.**
   Lag-1 averages +0.047 in the liquid half and +0.054 in the illiquid half — unlike
   crypto, the signal is not concentrated where it cannot be traded. Removing the
   capacity limit moves the edge only +1.39 → +2.15bp. The problem there is simply
   that ~2bp of tradable edge cannot pay a 25bp round trip.

4. **Equity reverse causality is weaker but not absent.** Mean reverse coefficient
   +0.019 against a forward +0.051, versus ≈0.000 reverse in crypto. The equity
   causality is real but less clean.

5. **An exploratory Hawkes finding did not replicate.** A crude decayed-jump proxy with
   full-sample thresholds appeared to split the loss-making tail into +19bp and −25bp
   halves. Under trailing-only thresholds, fitted parameters and purged folds, the gate
   is roughly neutral. The machinery works; the trading benefit was selection bias.
   This is exactly what the honest-inference layer was built to catch.

6. **The neural overlay adds nothing reliable.** Mean out-of-sample IC +0.023 on
   Solana (12/19 folds positive) but **−0.005** on equities (5/11 folds). In both cases
   the IC-shrunk blend behaves correctly — it declines to bet on a net that has not
   earned it.

## Two bugs worth knowing about

Both were caught by tests and both silently produced a near-empty panel rather than an
error, which is the dangerous failure mode:

- **Bar-grid detection.** US equity sessions open at 09:30, so hourly bars sit at :30
  past. A floor-to-the-hour grid check discarded 5,064 of 5,072 bars.
- **Timestamp units.** A parquet round trip returns millisecond-resolution timestamps;
  computing a within-interval offset as if they were nanoseconds produced nonsense.

Both are pinned by `tests/test_sessions.py`.

## Safety

There is **no live order routing**. `CcxtAdapter` requires `allow_live=True`, the
`STATARB_ALLOW_LIVE=1` environment variable, *and* credentials — and then still raises
`NotImplementedError`. Nothing here should be traded with real money: the backtest does
not show a profitable strategy.

## Data

Yahoo Finance hourly bars, 730-day rolling window, cached to parquet.

The equity follower list was produced by screening a broader candidate set on
**liquidity and data coverage only** — at least 4,500 bars, and average dollar volume
per bar between $30k and $3m — and keeping every survivor. Selection never referenced
the lead-lag coefficient being measured, so the basket is not chosen on the thing
being tested. CULP, the single strongest signal found during exploration (*t* = 3.81),
was dropped by that screen for being too thin, and stayed dropped.

Yahoo reports zero volume on ~half of hourly crypto bars, so all liquidity measures
use window aggregates rather than per-bar values.
