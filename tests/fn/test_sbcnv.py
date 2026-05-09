"""Tests for moirais.fn.sbcnv — bootstrap convergence diagnostic."""

import numpy as np
import pytest

from moirais.fn.sbcnv import sbcnv


def test_basic_output():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = sbcnv(x, n_reps=50, seed=7)
    assert "variances" in result
    assert "cv_ratio" in result


def test_normal_converges():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    result = sbcnv(x, n_reps=100, seed=1)
    assert result["converged"] is True or result["cv_ratio"] < 1.0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sbcnv(np.array([]))
