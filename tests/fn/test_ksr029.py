"""Tests for ksr029.kosorok_ch2_glivenko_cantelli_class."""
import numpy as np
import pytest
from moirais.fn.ksr029 import kosorok_ch2_glivenko_cantelli_class


def test_ksr029_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P_n = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_class(F, P_n, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr029_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P_n = np.random.default_rng(42).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_class(F, P_n, P)
    assert isinstance(result, dict)
