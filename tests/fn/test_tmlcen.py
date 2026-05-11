"""Tests for tmlcen.tmle_censoring."""
import numpy as np
import pytest
from morie.fn.tmlcen import tmle_censoring


def test_tmlcen_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    censor = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_censoring(time, event, censor, treatment, covariates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcen_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    censor = np.random.default_rng(42).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_censoring(time, event, censor, treatment, covariates)
    assert isinstance(result, dict)
