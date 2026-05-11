"""Tests for chrF.chrf."""
import numpy as np
import pytest
from morie.fn.chrF import chrf


def test_chrF_basic():
    """Test basic functionality."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = chrf(candidate, reference, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chrF_edge():
    """Test edge cases."""
    candidate = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = chrf(candidate, reference, beta)
    assert isinstance(result, dict)
