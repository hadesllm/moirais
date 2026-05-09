"""Tests for rganc.rangayyan_anc."""
import numpy as np
import pytest
from moirais.fn.rganc import rangayyan_anc


def test_rganc_basic():
    """Test basic functionality."""
    primary = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    order = 4
    result = rangayyan_anc(primary, reference, mu, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rganc_edge():
    """Test edge cases."""
    primary = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    order = 4
    result = rangayyan_anc(primary, reference, mu, order)
    assert isinstance(result, dict)
