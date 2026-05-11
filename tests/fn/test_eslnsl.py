"""Tests for eslnsl.esl_natural_spline."""
import numpy as np
import pytest
from morie.fn.eslnsl import esl_natural_spline


def test_eslnsl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    knots = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_natural_spline(x, knots)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslnsl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    knots = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_natural_spline(x, knots)
    assert isinstance(result, dict)
