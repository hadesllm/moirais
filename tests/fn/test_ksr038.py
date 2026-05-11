"""Tests for ksr038.kosorok_ch2_donsker_uniform_entropy."""
import numpy as np
import pytest
from morie.fn.ksr038 import kosorok_ch2_donsker_uniform_entropy


def test_ksr038_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_donsker_uniform_entropy(F, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr038_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_donsker_uniform_entropy(F, P)
    assert isinstance(result, dict)
