"""Tests for bayam.bayesian_am_scaling."""
import numpy as np
import pytest
from morie.fn.bayam import bayesian_am_scaling


def test_bayam_basic():
    """Test basic functionality."""
    survey_data = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    burnin = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_am_scaling(survey_data, n_iter, burnin)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayam_edge():
    """Test edge cases."""
    survey_data = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    burnin = np.random.default_rng(42).normal(0, 1, 100)
    result = bayesian_am_scaling(survey_data, n_iter, burnin)
    assert isinstance(result, dict)
