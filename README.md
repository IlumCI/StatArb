# StatArb — Lead-Lag Statistical Arbitrage on the Solana Ecosystem

A research-grade statistical arbitrage system built on the **lead-lag hypothesis**: a
liquid leader (SOL-USD) impounds information first, and illiquid followers in the same
ecosystem reprice with a delay. The system ranks a basket of illiquid followers by
predicted catch-up, filters out assets caught in self-exciting crash cascades using a
**Hawkes process**, and sizes positions under an explicit capacity constraint.

## The headline result

**The lead-lag effect is real, strong, and not tradable.**

Measured across 17,477 hourly bars per asset over two years:

| | |
|---|---|
| Gross edge, ignoring capacity | **+8.50 bp/trade**, clustered *t* = **4.22** |
| Gross edge, respecting capacity | **−0.14 bp/trade**, clustered *t* = **−0.04** |
| Round-trip cost | ~47 bp |
| Net Sharpe | ≈ −8 |

The entire measured edge lives in assets that cannot be filled. The strongest signals
sit on tokens turning over a few dollars an hour — a $12.5k position in MOBILE-USD
would be **74,000×** its average hourly volume. Once positions are capped at 10% of
bar volume, the edge vanishes into noise.

That is an economically coherent result rather than a disappointing one: the
inefficiency persists *because* it cannot be arbitraged. See
[`artifacts/report.md`](artifacts/report.md) for the full write-up.

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

statarb fetch                       # populate the bar cache
statarb research leadlag            # cross-correlation by liquidity tier
statarb research hawkes             # fitted branching ratios per asset
statarb backtest --construction both --notional 10000
statarb capacity                    # net edge vs book size
statarb paper --once --dry-run      # one live cycle, logs intents only
```

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
taker fees. Corwin-Schultz is computed but deliberately *not* used for charging: it
estimates near-zero spreads on thin crypto OHLC, which would flatter the backtest
exactly where reality is worst.

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

1. **Lead-lag is ordered by illiquidity.** Lag-1 correlation rises monotonically as
   liquidity falls (SAMO 0.108 *t*=14.4 → JTO 0.010 *t*=1.4) while lag-0 falls in step,
   and reverse causality is ≈0 everywhere. This is a genuine lead-lag, not co-movement.

2. **The edge is entirely a capacity illusion.** +8.50bp ignoring capacity, −0.14bp
   respecting it.

3. **An exploratory Hawkes finding did not replicate.** A crude decayed-jump proxy with
   full-sample thresholds appeared to split the loss-making tail into +19bp and −25bp
   halves. Under trailing-only thresholds, fitted parameters and purged folds, the gate
   is roughly neutral. The machinery works; the trading benefit was selection bias.
   This is exactly what the honest-inference layer was built to catch.

4. **The neural overlay adds a little, not significantly.** Mean out-of-sample IC
   +0.023 across 19 folds (12 positive), lifting gross Sharpe 0.49 → 0.66. Well within
   noise, and the IC-shrunk blend correctly declines to bet on it.

## Safety

There is **no live order routing**. `CcxtAdapter` requires `allow_live=True`, the
`STATARB_ALLOW_LIVE=1` environment variable, *and* credentials — and then still raises
`NotImplementedError`. Nothing here should be traded with real money: the backtest does
not show a profitable strategy.

## Data

Yahoo Finance hourly bars, 730-day rolling window, cached to parquet. Note that Yahoo
reports zero volume on ~half of hourly crypto bars, so all liquidity measures use
window aggregates rather than per-bar values.
