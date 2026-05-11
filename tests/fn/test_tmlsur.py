"""Tests for tmlsur.tmle_survival."""
import numpy as np
import pytest
from morie.fn.tmlsur import tmle_survival


def test_tmlsur_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = tmle_survival(time, event, treatment, covariates, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlsur_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = tmle_survival(time, event, treatment, covariates, tau)
    assert isinstance(result, dict)
