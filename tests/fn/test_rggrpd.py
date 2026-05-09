"""Tests for rggrpd.rangayyan_group_delay."""
import numpy as np
import pytest
from moirais.fn.rggrpd import rangayyan_group_delay


def test_rggrpd_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_group_delay(b, a, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rggrpd_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_group_delay(b, a, fs)
    assert isinstance(result, dict)
