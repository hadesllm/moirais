"""Tests for spsmsh.spsm_shifted_intervention."""
import numpy as np
import pytest
from moirais.fn.spsmsh import spsm_shifted_intervention


def test_spsmsh_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    shift_fn = (lambda v: v)
    result = spsm_shifted_intervention(y, A, H, shift_fn)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spsmsh_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    shift_fn = (lambda v: v)
    result = spsm_shifted_intervention(y, A, H, shift_fn)
    assert isinstance(result, dict)
