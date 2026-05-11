"""Tests for ksr037.kosorok_ch2_glivenko_cantelli_uniform."""
import numpy as np
import pytest
from morie.fn.ksr037 import kosorok_ch2_glivenko_cantelli_uniform


def test_ksr037_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_uniform(F, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr037_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_glivenko_cantelli_uniform(F, P)
    assert isinstance(result, dict)
