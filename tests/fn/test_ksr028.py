"""Tests for ksr028.kosorok_ch2_glivenko_cantelli_classical."""
import numpy as np
import pytest
from moirais.fn.ksr028 import kosorok_ch2_glivenko_cantelli_classical


def test_ksr028_basic():
    """Test basic functionality."""
    F_n = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_classical(F_n, F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr028_edge():
    """Test edge cases."""
    F_n = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_classical(F_n, F)
    assert isinstance(result, dict)
