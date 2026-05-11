"""Tests for morie.fn.pnreg — Penalized kernel regression."""

import numpy as np
import pytest
from morie.fn.pnreg import pnreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 50)
    result = pnreg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "bandwidth", "penalty", "n_obs"):
        assert key in result


def test_penalty_effect():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.5, 50)
    r1 = pnreg(x, y, penalty=0.01)
    r2 = pnreg(x, y, penalty=100.0)
    v1 = np.var(r1["y_hat"])
    v2 = np.var(r2["y_hat"])
    assert v2 <= v1 + 1e-6


def test_negative_penalty_raises():
    with pytest.raises(ValueError, match="penalty"):
        pnreg(np.ones(10), np.ones(10), penalty=-1)


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 3"):
        pnreg(np.ones(2), np.ones(2))
