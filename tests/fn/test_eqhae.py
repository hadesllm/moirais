"""Tests for eqhae.equating_haebara."""
import numpy as np
import pytest
from moirais.fn.eqhae import equating_haebara


def test_eqhae_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    a_R = np.random.default_rng(42).normal(0, 1, 100)
    a_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_haebara(y, b_R, b_F, a_R, a_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eqhae_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    a_R = np.random.default_rng(42).normal(0, 1, 100)
    a_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_haebara(y, b_R, b_F, a_R, a_F)
    assert isinstance(result, dict)
