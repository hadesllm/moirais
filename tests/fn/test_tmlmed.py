"""Tests for tmlmed.tmle_mediation."""
import numpy as np
import pytest
from morie.fn.tmlmed import tmle_mediation


def test_tmlmed_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    mediator = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_mediation(y, treatment, mediator, covariates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmed_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treatment = np.random.default_rng(42).normal(0, 1, 100)
    mediator = np.random.default_rng(42).normal(0, 1, 100)
    covariates = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_mediation(y, treatment, mediator, covariates)
    assert isinstance(result, dict)
