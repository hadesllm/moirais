"""Tests for bhltmsm.behavioral_health_msm."""
import numpy as np
import pytest
from moirais.fn.bhltmsm import behavioral_health_msm


def test_bhltmsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = behavioral_health_msm(y, A, H, baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bhltmsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = behavioral_health_msm(y, A, H, baseline)
    assert isinstance(result, dict)
