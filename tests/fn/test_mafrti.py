"""Tests for mafrti.ma_freeman_tukey_inverse."""
import numpy as np
import pytest
from moirais.fn.mafrti import ma_freeman_tukey_inverse


def test_mafrti_basic():
    """Test basic functionality."""
    ft = np.random.default_rng(42).normal(0, 1, 100)
    n_harmonic = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_freeman_tukey_inverse(ft, n_harmonic)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mafrti_edge():
    """Test edge cases."""
    ft = np.random.default_rng(42).normal(0, 1, 100)
    n_harmonic = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_freeman_tukey_inverse(ft, n_harmonic)
    assert isinstance(result, dict)
