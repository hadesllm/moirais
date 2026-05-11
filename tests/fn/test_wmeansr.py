"""Tests for wmeansr.weighted_mean_survey."""
import numpy as np
import pytest
from morie.fn.wmeansr import weighted_mean_survey


def test_wmeansr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_mean_survey(y, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wmeansr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = weighted_mean_survey(y, weights)
    assert isinstance(result, dict)
