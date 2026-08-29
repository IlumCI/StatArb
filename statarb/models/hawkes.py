"""Hawkes self-exciting process for crash-cascade detection.

Why this matters here. The lead-lag edge is strongly positive on ordinary large
leader moves but flips sharply negative in the extreme tail, because some of those
moves coincide with a follower already in a self-reinforcing sell-off. Such a
follower does not "catch up" to the leader; it keeps falling. Splitting the tail by
conditional jump intensity separates the two cases, and empirically that split is
worth far more than the leader-move threshold on its own.

The model is a univariate exponential-kernel Hawkes process on *negative* jump
arrivals,

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

fitted by maximum likelihood. ``alpha / beta`` is the branching ratio: the expected
number of aftershocks each jump directly triggers. Below 1 the process is stationary;
as it approaches 1 the asset is one shock away from a cascade.

A bivariate extension lets the *leader's* jumps excite the follower, which is the
lead-lag hypothesis expressed in jump space rather than in returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

_EPS = 1e-12


@dataclass(frozen=True)
class HawkesParams:
    """Fitted exponential-kernel parameters, in events-per-bar units."""

    mu: float
    alpha: float
    beta: float
    log_likelihood: float
    n_events: int
    converged: bool

    @property
    def branching_ratio(self) -> float:
        """Expected direct offspring per event. >= 1 means an explosive process."""
        return float(self.alpha / self.beta) if self.beta > _EPS else float("inf")

    @property
    def stationary(self) -> bool:
        return self.branching_ratio < 1.0

    @property
    def expected_cascade_size(self) -> float:
        """Total expected events triggered by one exogenous shock, 1/(1-n)."""
        n = self.branching_ratio
        return float(1.0 / (1.0 - n)) if n < 1.0 else float("inf")

    @property
    def half_life(self) -> float:
        """Bars for the excitation from one event to decay by half."""
        return float(np.log(2.0) / self.beta) if self.beta > _EPS else float("inf")


def _recursion(events: np.ndarray, beta: float) -> np.ndarray:
    """A_i = sum_{j<i} exp(-beta (t_i - t_j)), computed in O(n).

    The naive double sum is O(n^2) and would make refitting across 19 folds and 14
    assets impractical; the exponential kernel is what permits the recursion.
    """
    n = len(events)
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = np.exp(-beta * (events[i] - events[i - 1])) * (1.0 + a[i - 1])
    return a


def _neg_log_likelihood(theta: np.ndarray, events: np.ndarray, horizon: float) -> float:
    """Negative log-likelihood, parameterised in log space to keep params positive."""
    mu, alpha, beta = np.exp(theta)
    if not np.isfinite([mu, alpha, beta]).all() or beta <= _EPS:
        return 1e12
    a = _recursion(events, beta)
    intensity = mu + alpha * a
    if np.any(intensity <= 0):
        return 1e12
    # Compensator: integral of lambda over [0, T].
    compensator = mu * horizon + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (horizon - events)))
    ll = np.sum(np.log(intensity)) - compensator
    return -ll if np.isfinite(ll) else 1e12


def fit_hawkes(
    events: np.ndarray,
    horizon: float,
    *,
    mu_bounds: tuple[float, float] = (1e-6, 10.0),
    alpha_bounds: tuple[float, float] = (1e-6, 50.0),
    beta_bounds: tuple[float, float] = (1e-4, 50.0),
    n_restarts: int = 3,
    seed: int = 0,
) -> HawkesParams:
    """Fit by maximum likelihood with multiple restarts.

    The likelihood is multi-modal in ``beta``, so a single start from a fixed guess
    lands in a local optimum often enough to matter. Restarts are seeded
    deterministically so a backtest stays reproducible.
    """
    events = np.asarray(events, dtype="float64")
    events = np.sort(events[np.isfinite(events)])
    n = len(events)
    if n < 10 or horizon <= 0:
        # Too few events to identify excitation; fall back to a Poisson process.
        rate = max(n / horizon, 1e-6) if horizon > 0 else 1e-6
        return HawkesParams(rate, 1e-6, 1.0, float("nan"), n, converged=False)

    rate = n / horizon
    rng = np.random.default_rng(seed)
    starts = [np.log([rate * 0.5, 0.5, 1.0]), np.log([rate * 0.8, 0.2, 0.5]), np.log([rate * 0.3, 1.0, 2.0])]
    for _ in range(max(0, n_restarts - len(starts))):
        starts.append(np.log([rate * rng.uniform(0.2, 0.9), rng.uniform(0.1, 1.5), rng.uniform(0.3, 3.0)]))

    bounds = [
        (np.log(mu_bounds[0]), np.log(mu_bounds[1])),
        (np.log(alpha_bounds[0]), np.log(alpha_bounds[1])),
        (np.log(beta_bounds[0]), np.log(beta_bounds[1])),
    ]
    best: optimize.OptimizeResult | None = None
    for start in starts[:n_restarts]:
        try:
            res = optimize.minimize(
                _neg_log_likelihood, start, args=(events, horizon),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-10},
            )
        except (ValueError, FloatingPointError):  # pragma: no cover - optimiser edge case
            continue
        if res.fun < 1e11 and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return HawkesParams(rate, 1e-6, 1.0, float("nan"), n, converged=False)
    mu, alpha, beta = np.exp(best.x)
    return HawkesParams(float(mu), float(alpha), float(beta), float(-best.fun), n, bool(best.success))


def conditional_intensity(
    params: HawkesParams, events: np.ndarray, eval_times: np.ndarray
) -> np.ndarray:
    """Evaluate lambda(t) at arbitrary times using only strictly earlier events.

    The strict inequality is deliberate: at the moment a jump prints, the decision
    must not already incorporate that jump's own excitation.
    """
    events = np.sort(np.asarray(events, dtype="float64"))
    eval_times = np.asarray(eval_times, dtype="float64")
    out = np.full(len(eval_times), params.mu, dtype="float64")
    if len(events) == 0:
        return out

    order = np.argsort(eval_times)
    # Walk both sorted sequences once, carrying the decayed excitation forward.
    excitation, last_t, j = 0.0, None, 0
    for pos in order:
        t = eval_times[pos]
        while j < len(events) and events[j] < t:
            if last_t is not None:
                excitation *= np.exp(-params.beta * (events[j] - last_t))
            last_t = events[j]
            excitation += params.alpha
            j += 1
        if last_t is None:
            out[pos] = params.mu
        else:
            out[pos] = params.mu + excitation * np.exp(-params.beta * (t - last_t))
    return out


@dataclass(frozen=True)
class BivariateHawkesParams:
    """Follower intensity excited by both its own jumps and the leader's."""

    mu: float
    alpha_self: float
    alpha_cross: float
    beta: float
    log_likelihood: float
    n_events: int
    converged: bool

    @property
    def self_branching(self) -> float:
        return float(self.alpha_self / self.beta) if self.beta > _EPS else float("inf")

    @property
    def cross_branching(self) -> float:
        """Expected follower jumps triggered per leader jump: lead-lag in jump space."""
        return float(self.alpha_cross / self.beta) if self.beta > _EPS else float("inf")


def _cross_excitation(own: np.ndarray, cross: np.ndarray, beta: float) -> np.ndarray:
    """sum_{c_j < t_i} exp(-beta (t_i - c_j)) for each own-event time, in O(n + m).

    Carries the decayed excitation forward through a merged walk of both sequences so
    every exponent is a non-positive time difference. Computing it as a rescaled
    cumulative sum instead overflows: the rescaling factor is exp(+beta * T).
    """
    out = np.zeros(len(own))
    excitation, last, j = 0.0, None, 0
    for i, t in enumerate(own):
        while j < len(cross) and cross[j] < t:
            if last is not None:
                excitation *= np.exp(-beta * (cross[j] - last))
            last = cross[j]
            excitation += 1.0
            j += 1
        out[i] = excitation * np.exp(-beta * (t - last)) if last is not None else 0.0
    return out


def _bivariate_nll(
    theta: np.ndarray, own: np.ndarray, cross: np.ndarray, horizon: float
) -> float:
    mu, a_self, a_cross, beta = np.exp(theta)
    if beta <= _EPS or not np.isfinite([mu, a_self, a_cross, beta]).all():
        return 1e12
    intensity = mu + a_self * _recursion(own, beta) + a_cross * _cross_excitation(own, cross, beta)
    if np.any(intensity <= 0) or not np.isfinite(intensity).all():
        return 1e12
    comp = (
        mu * horizon
        + (a_self / beta) * np.sum(1.0 - np.exp(-beta * (horizon - own)))
        + (a_cross / beta) * np.sum(1.0 - np.exp(-beta * (horizon - cross)))
    )
    ll = np.sum(np.log(intensity)) - comp
    return -ll if np.isfinite(ll) else 1e12


def fit_bivariate_hawkes(
    own_events: np.ndarray, cross_events: np.ndarray, horizon: float, *, seed: int = 0
) -> BivariateHawkesParams:
    """Fit a follower's jump intensity with leader cross-excitation.

    A large ``cross_branching`` means the leader's crashes propagate into this
    follower, which is precisely the mechanism the strategy is betting on, and also
    precisely what makes the follower dangerous to hold during a cascade.
    """
    own = np.sort(np.asarray(own_events, dtype="float64"))
    cross = np.sort(np.asarray(cross_events, dtype="float64"))
    n = len(own)
    if n < 10 or horizon <= 0:
        rate = max(n / horizon, 1e-6) if horizon > 0 else 1e-6
        return BivariateHawkesParams(rate, 1e-6, 1e-6, 1.0, float("nan"), n, False)

    rate = n / horizon
    starts = [
        np.log([rate * 0.6, 0.2, 0.2, 1.0]),
        np.log([rate * 0.4, 0.5, 0.1, 0.3]),
        np.log([rate * 0.8, 0.1, 0.5, 2.0]),
    ]
    bounds = [(np.log(1e-6), np.log(10.0)), (np.log(1e-6), np.log(50.0)),
              (np.log(1e-6), np.log(50.0)), (np.log(1e-4), np.log(50.0))]
    best = None
    for start in starts:
        try:
            res = optimize.minimize(
                _bivariate_nll, start, args=(own, cross, horizon),
                method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-10},
            )
        except (ValueError, FloatingPointError):  # pragma: no cover
            continue
        if res.fun < 1e11 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return BivariateHawkesParams(rate, 1e-6, 1e-6, 1.0, float("nan"), n, False)
    res = best
    mu, a_self, a_cross, beta = np.exp(res.x)
    return BivariateHawkesParams(
        float(mu), float(a_self), float(a_cross), float(beta), float(-res.fun), n, bool(res.success)
    )


def simulate_hawkes(
    mu: float, alpha: float, beta: float, horizon: float, *, seed: int = 0
) -> np.ndarray:
    """Simulate by Ogata's thinning algorithm. Used to validate the estimator."""
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    while t < horizon:
        # Upper bound on lambda over [t, .): intensity only decays between events.
        lam_bar = mu + alpha * np.sum(np.exp(-beta * (t - np.asarray(events)))) if events else mu
        if lam_bar <= 0:
            break
        t += rng.exponential(1.0 / lam_bar)
        if t >= horizon:
            break
        lam = mu + (alpha * np.sum(np.exp(-beta * (t - np.asarray(events)))) if events else 0.0)
        if rng.random() <= lam / lam_bar:
            events.append(t)
    return np.asarray(events)


def jump_events(
    returns: pd.Series, threshold: pd.Series, *, negative_only: bool = True
) -> np.ndarray:
    """Bar positions where a jump occurred, as float times in bar units.

    ``threshold`` must already be a trailing, shifted quantile so that whether a bar
    counts as a jump is decided without reference to the future.
    """
    aligned = pd.concat([returns.rename("r"), threshold.rename("q")], axis=1)
    hit = (aligned["r"] < -aligned["q"]) if negative_only else (aligned["r"].abs() > aligned["q"])
    hit = hit & aligned["q"].notna() & aligned["r"].notna()
    return np.flatnonzero(hit.to_numpy())
