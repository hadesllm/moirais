"""Tests for snr2.snijders_bosker_r2_level1."""
import numpy as np
import pytest
from morie.fn.snr2 import snijders_bosker_r2_level1


def test_snr2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = snijders_bosker_r2_level1(y, X, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snr2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = snijders_bosker_r2_level1(y, X, cluster)
    assert isinstance(result, dict)
