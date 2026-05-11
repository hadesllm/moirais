"""Tests for morie.fn.coxph -- Cox proportional hazards model."""

import numpy as np
import pytest

from morie.fn.coxph import coxph


@pytest.fixture()
def cox_data():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.standard_normal(n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    X = np.column_stack([x1, x2])
    lp = 0.5 * x1 - 0.3 * x2
    t_true = rng.exponential(np.exp(-lp))
    c = rng.exponential(3, n)
    time = np.minimum(t_true, c)
    event = (t_true <= c).astype(float)
    return time, event, X


def test_returns_dict(cox_data):
    time, event, X = cox_data
    result = coxph(time, event, X)
    assert isinstance(result, dict)
    for k in ("coefficients", "se", "hazard_ratios", "z_scores",
              "p_values", "log_partial_likelihood", "n_iter",
              "converged", "n_obs", "n_events"):
        assert k in result


def test_coefficients_shape(cox_data):
    time, event, X = cox_data
    result = coxph(time, event, X)
    assert result["coefficients"].shape == (2,)


def test_hazard_ratios_positive(cox_data):
    time, event, X = cox_data
    result = coxph(time, event, X)
    assert np.all(result["hazard_ratios"] > 0)


def test_se_positive(cox_data):
    time, event, X = cox_data
    result = coxph(time, event, X)
    assert np.all(result["se"] > 0)


def test_converges(cox_data):
    time, event, X = cox_data
    result = coxph(time, event, X)
    assert result["converged"]


def test_n_events_correct(cox_data):
    time, event, X = cox_data
    result = coxph(time, event, X)
    assert result["n_events"] == int(np.sum(event))


def test_dimension_error():
    with pytest.raises(ValueError):
        coxph(np.array([1, 2]), np.array([1, 1, 1]), np.array([[1], [2]]))


def test_cheatsheet():
    from morie.fn.coxph import cheatsheet
    assert "cox" in cheatsheet().lower()
