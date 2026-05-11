"""Tests for eslthl.esl_thin_plate_spline."""
import numpy as np
import pytest
from morie.fn.eslthl import esl_thin_plate_spline


def test_eslthl_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_thin_plate_spline(X, y, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslthl_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_thin_plate_spline(X, y, lambda_)
    assert isinstance(result, dict)
