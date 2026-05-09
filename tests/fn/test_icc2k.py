"""Tests for icc2k.icc_two_way_random_avg."""
import numpy as np
import pytest
from moirais.fn.icc2k import icc_two_way_random_avg


def test_icc2k_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_two_way_random_avg(y, subject, rater)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icc2k_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_two_way_random_avg(y, subject, rater)
    assert isinstance(result, dict)
