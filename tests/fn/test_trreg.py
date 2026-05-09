"""Tests for moirais.fn.trreg — transformation regression."""

import numpy as np
import pytest

from moirais.fn.trreg import trreg


def test_log_transform():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = np.exp(1.0 + 0.5 * X[:, 0] + 0.3 * rng.standard_normal(n))
    result = trreg(y, X, transform="log")
    assert result["lambda_"] == 0.0
    assert result["beta"][1] == pytest.approx(0.5, abs=0.3)


def test_box_cox():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 1))
    y = np.exp(1.0 + 0.5 * X[:, 0] + rng.standard_normal(n) * 0.3)
    result = trreg(y, X, transform="box-cox")
    assert "lambda_" in result


def test_nonpositive_raises():
    with pytest.raises(ValueError, match="positive"):
        trreg(np.array([-1.0, 2.0]), np.array([[1.0], [2.0]]))
