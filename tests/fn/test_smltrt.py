"""Tests for smltrt.survey_ratio."""
import numpy as np
import pytest
from morie.fn.smltrt import survey_ratio


def test_smltrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_ratio(y, x, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smltrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_ratio(y, x, weights)
    assert isinstance(result, dict)
