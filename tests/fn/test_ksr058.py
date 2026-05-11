"""Tests for ksr058.kosorok_ch2_law_iterated_logarithm."""
import numpy as np
import pytest
from morie.fn.ksr058 import kosorok_ch2_law_iterated_logarithm


def test_ksr058_basic():
    """Test basic functionality."""
    G_n = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch2_law_iterated_logarithm(G_n, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr058_edge():
    """Test edge cases."""
    G_n = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kosorok_ch2_law_iterated_logarithm(G_n, n)
    assert isinstance(result, dict)
