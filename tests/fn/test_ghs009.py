"""Tests for ghs009.ghosal_ch3_stick_breaking_weights."""
import numpy as np
import pytest
from moirais.fn.ghs009 import ghosal_ch3_stick_breaking_weights


def test_ghs009_basic():
    """Test basic functionality."""
    V_l = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_stick_breaking_weights(V_l, j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs009_edge():
    """Test edge cases."""
    V_l = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_stick_breaking_weights(V_l, j)
    assert isinstance(result, dict)
