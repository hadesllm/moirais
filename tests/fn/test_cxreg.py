"""Tests for moirais.fn.cxreg — Cox regression."""

import numpy as np
import pytest

from moirais.fn.cxreg import cxreg


@pytest.fixture()
def surv_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    true_beta = np.array([0.5, -0.3])
    hazard = np.exp(X @ true_beta)
    time = rng.exponential(1.0 / hazard)
    censor = rng.exponential(3.0, size=n)
    event = (time <= censor).astype(float)
    time = np.minimum(time, censor)
    return time, event, X


def test_basic_output(surv_data):
    time, event, X = surv_data
    result = cxreg(time, event, X)
    assert "beta" in result
    assert "hr" in result
    assert result["n"] == 200


def test_beta_direction(surv_data):
    time, event, X = surv_data
    result = cxreg(time, event, X)
    assert result["beta"][0] > 0


def test_hazard_ratios(surv_data):
    time, event, X = surv_data
    result = cxreg(time, event, X)
    np.testing.assert_allclose(result["hr"], np.exp(result["beta"]))


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cxreg(np.array([]), np.array([]), np.array([]).reshape(0, 1))
