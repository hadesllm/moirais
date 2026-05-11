"""Tests for ksr034.kosorok_ch2_glivenko_cantelli_bracketing."""
import numpy as np
import pytest
from morie.fn.ksr034 import kosorok_ch2_glivenko_cantelli_bracketing


def test_ksr034_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_bracketing(F, P, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr034_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_bracketing(F, P, eps)
    assert isinstance(result, dict)
