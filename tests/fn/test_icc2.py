"""Tests for icc2.icc_two_way_random."""
import numpy as np
import pytest
from moirais.fn.icc2 import icc_two_way_random


def test_icc2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_two_way_random(y, subject, rater)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icc2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_two_way_random(y, subject, rater)
    assert isinstance(result, dict)
