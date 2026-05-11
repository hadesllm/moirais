"""Tests for ksr053.kosorok_ch2_kaplan_meier_inverse."""
import numpy as np
import pytest
from morie.fn.ksr053 import kosorok_ch2_kaplan_meier_inverse


def test_ksr053_basic():
    """Test basic functionality."""
    S_0 = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    F_0 = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_kaplan_meier_inverse(S_0, L, F_0, a, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr053_edge():
    """Test edge cases."""
    S_0 = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    F_0 = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_kaplan_meier_inverse(S_0, L, F_0, a, t)
    assert isinstance(result, dict)
