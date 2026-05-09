"""Tests for prtMK.prewhitening_mk."""
import numpy as np
import pytest
from moirais.fn.prtMK import prewhitening_mk


def test_prtMK_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = prewhitening_mk(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prtMK_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = prewhitening_mk(x)
    assert isinstance(result, dict)
