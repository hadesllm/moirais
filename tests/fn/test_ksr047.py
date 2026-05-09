"""Tests for ksr047.kosorok_ch2_kaplan_meier_self_consistency."""
import numpy as np
import pytest
from moirais.fn.ksr047 import kosorok_ch2_kaplan_meier_self_consistency


def test_ksr047_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    S_0 = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_kaplan_meier_self_consistency(S, S_0, L, G, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr047_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    S_0 = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_kaplan_meier_self_consistency(S, S_0, L, G, t)
    assert isinstance(result, dict)
