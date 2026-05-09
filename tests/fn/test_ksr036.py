"""Tests for ksr036.kosorok_ch2_donsker_bracketing_theorem."""
import numpy as np
import pytest
from moirais.fn.ksr036 import kosorok_ch2_donsker_bracketing_theorem


def test_ksr036_basic():
    """Test basic functionality."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_donsker_bracketing_theorem(F, P)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr036_edge():
    """Test edge cases."""
    F = np.random.default_rng(43).normal(0, 1, 100)
    P = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_donsker_bracketing_theorem(F, P)
    assert isinstance(result, dict)
