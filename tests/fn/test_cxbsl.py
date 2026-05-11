"""Tests for morie.fn.cxbsl — Breslow baseline hazard."""

import numpy as np
import pytest

from morie.fn.cxbsl import cxbsl


def test_basic_output():
    rng = np.random.default_rng(42)
    n = 100
    time = rng.exponential(1.0, size=n)
    event = rng.binomial(1, 0.7, size=n).astype(float)
    X = rng.standard_normal((n, 1))
    beta = np.array([0.5])
    result = cxbsl(time, event, X, beta)
    assert "cumhazard" in result
    assert "baseline_survival" in result


def test_cumhazard_nondecreasing():
    rng = np.random.default_rng(7)
    n = 200
    time = rng.exponential(1.0, size=n)
    event = np.ones(n)
    X = rng.standard_normal((n, 1))
    beta = np.array([0.0])
    result = cxbsl(time, event, X, beta)
    assert np.all(np.diff(result["cumhazard"]) >= -1e-10)


def test_survival_between_01():
    rng = np.random.default_rng(42)
    n = 100
    time = rng.exponential(1.0, size=n)
    event = np.ones(n)
    X = rng.standard_normal((n, 1))
    beta = np.array([0.0])
    result = cxbsl(time, event, X, beta)
    assert np.all(result["baseline_survival"] >= 0)
    assert np.all(result["baseline_survival"] <= 1)
