"""Hawkes estimator validation."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.models.hawkes import (
    HawkesParams,
    conditional_intensity,
    fit_bivariate_hawkes,
    fit_hawkes,
    simulate_hawkes,
)


@pytest.mark.parametrize(
    ("mu", "alpha", "beta"),
    [(0.05, 0.30, 0.60), (0.02, 0.80, 1.20), (0.10, 0.15, 0.50)],
)
def test_mle_recovers_simulated_parameters(mu, alpha, beta):
    """Simulate a process with known parameters and check the fit recovers them."""
    events = simulate_hawkes(mu, alpha, beta, horizon=20000, seed=3)
    assert len(events) > 500
    fitted = fit_hawkes(events, 20000.0, seed=1)
    true_branching = alpha / beta
    assert fitted.converged
    # The branching ratio is the quantity the risk layer actually consumes, so it is
    # what the tolerance is set on.
    assert abs(fitted.branching_ratio - true_branching) < 0.08
    assert abs(fitted.mu - mu) < 0.02


def test_branching_ratio_and_cascade_size():
    p = HawkesParams(mu=0.1, alpha=0.5, beta=1.0, log_likelihood=0.0, n_events=100, converged=True)
    assert p.branching_ratio == pytest.approx(0.5)
    assert p.expected_cascade_size == pytest.approx(2.0)
    assert p.stationary
    explosive = HawkesParams(0.1, 2.0, 1.0, 0.0, 100, True)
    assert not explosive.stationary
    assert explosive.expected_cascade_size == float("inf")


def test_intensity_rises_after_events_and_decays():
    p = HawkesParams(mu=0.1, alpha=1.0, beta=0.5, log_likelihood=0.0, n_events=3, converged=True)
    events = np.array([10.0, 10.5, 11.0])
    lam = conditional_intensity(p, events, np.array([9.0, 11.1, 20.0, 100.0]))
    assert lam[0] == pytest.approx(0.1)          # before any event: baseline only
    assert lam[1] > 1.0                           # just after a cluster: strongly excited
    assert lam[2] < lam[1]                        # decays with time
    assert lam[3] == pytest.approx(0.1, abs=1e-6)  # long after: back to baseline


def test_intensity_excludes_simultaneous_event():
    """A jump must not excite the very bar it prints on: that would be lookahead."""
    p = HawkesParams(mu=0.1, alpha=1.0, beta=0.5, log_likelihood=0.0, n_events=1, converged=True)
    lam = conditional_intensity(p, np.array([10.0]), np.array([10.0]))
    assert lam[0] == pytest.approx(0.1)


def test_bivariate_recovers_cross_excitation():
    """Leader jumps that trigger follower jumps should show up as cross-excitation."""
    rng = np.random.default_rng(5)
    lead = simulate_hawkes(0.03, 0.2, 0.5, horizon=20000, seed=2)
    own = list(simulate_hawkes(0.01, 0.1, 0.5, horizon=20000, seed=4))
    for t in lead:
        for _ in range(rng.poisson(0.4)):
            own.append(t + rng.exponential(2.0))
    own = np.sort(np.array([t for t in own if t < 20000]))
    fitted = fit_bivariate_hawkes(own, lead, 20000.0)
    assert fitted.converged
    assert 0.15 < fitted.cross_branching < 0.75


def test_too_few_events_falls_back_to_poisson():
    fitted = fit_hawkes(np.array([1.0, 5.0]), 100.0)
    assert not fitted.converged
    assert fitted.branching_ratio < 1e-3
