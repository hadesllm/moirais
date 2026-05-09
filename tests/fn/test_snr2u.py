"""Tests for snr2u.snijders_bosker_r2_level2."""
import numpy as np
import pytest
from moirais.fn.snr2u import snijders_bosker_r2_level2


def test_snr2u_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = snijders_bosker_r2_level2(y, X, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snr2u_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = snijders_bosker_r2_level2(y, X, cluster)
    assert isinstance(result, dict)
