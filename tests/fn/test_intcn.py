"""Tests for morie.fn.intcn -- Interval censoring (Turnbull)."""

import numpy as np
import pytest

from morie.fn.intcn import intcn


def test_returns_dict():
    left = np.array([1, 2, 3, 4, 5.0])
    right = np.array([2, 3, 4, 5, 6.0])
    r = intcn(left, right)
    assert isinstance(r, dict)
    for k in ("times", "survival", "mass", "n_obs", "converged"):
        assert k in r


def test_mass_sums_to_one():
    left = np.array([1, 2, 3, 4, 5.0])
    right = np.array([2, 3, 4, 5, 6.0])
    r = intcn(left, right)
    assert abs(np.sum(r["mass"]) - 1.0) < 1e-6


def test_survival_monotone():
    rng = np.random.default_rng(42)
    left = rng.uniform(0, 10, 50)
    right = left + rng.uniform(0.5, 2, 50)
    r = intcn(left, right)
    assert np.all(np.diff(r["survival"]) <= 1e-10)


def test_right_censored():
    left = np.array([1, 2, 3.0])
    right = np.array([2, np.inf, np.inf])
    r = intcn(left, right)
    assert r["n_obs"] == 3


def test_cheatsheet():
    from morie.fn.intcn import cheatsheet
    assert "interval" in cheatsheet().lower()
