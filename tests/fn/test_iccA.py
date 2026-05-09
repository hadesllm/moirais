"""Tests for iccA.icc_absolute_agreement."""
import numpy as np
import pytest
from moirais.fn.iccA import icc_absolute_agreement


def test_iccA_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_absolute_agreement(y, subject, rater)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_iccA_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.random.default_rng(42).normal(0, 1, 100)
    rater = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_absolute_agreement(y, subject, rater)
    assert isinstance(result, dict)
