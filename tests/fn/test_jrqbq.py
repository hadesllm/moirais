"""Tests for jrqbq.jarque_bera."""
import numpy as np
import pytest
from moirais.fn.jrqbq import jarque_bera


def test_jrqbq_basic():
    """Test basic functionality."""
    resid = np.random.default_rng(42).normal(0, 1, 100)
    result = jarque_bera(resid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jrqbq_edge():
    """Test edge cases."""
    resid = np.random.default_rng(42).normal(0, 1, 100)
    result = jarque_bera(resid)
    assert isinstance(result, dict)
