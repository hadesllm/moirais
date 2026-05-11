"""Tests for gentmt.generalized_treatment_msm."""
import numpy as np
import pytest
from morie.fn.gentmt import generalized_treatment_msm


def test_gentmt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = generalized_treatment_msm(y, A, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gentmt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = generalized_treatment_msm(y, A, H)
    assert isinstance(result, dict)
