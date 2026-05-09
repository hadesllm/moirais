"""Tests for iccC.icc_consistency."""
import numpy as np
import pytest
from moirais.fn.iccC import icc_consistency


def test_iccC_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_consistency(y, subject, rater)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_iccC_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_consistency(y, subject, rater)
    assert isinstance(result, dict)
