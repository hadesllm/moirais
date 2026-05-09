"""Tests for moirais.fn.bstep -- Bayesian change-point."""

import numpy as np
from moirais.fn.bstep import bayesian_changepoint


def test_returns_dict():
    data = np.concatenate([np.zeros(50), np.ones(50) * 5])
    result = bayesian_changepoint(data)
    assert isinstance(result, dict)
    assert "change_points" in result


def test_detects_changepoint():
    data = np.concatenate([np.zeros(50), np.ones(50) * 10])
    result = bayesian_changepoint(data)
    assert result["n_changepoints"] >= 1


def test_no_changepoint_for_constant():
    data = np.ones(100) * 3
    result = bayesian_changepoint(data)
    assert result["n_changepoints"] == 0
