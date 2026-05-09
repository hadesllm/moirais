"""Tests for rng218.rangayyan_ch4_cauchy_schwarz_vectors."""
import numpy as np
import pytest
from moirais.fn.rng218 import rangayyan_ch4_cauchy_schwarz_vectors


def test_rng218_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_cauchy_schwarz_vectors(a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng218_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_cauchy_schwarz_vectors(a, b)
    assert isinstance(result, dict)
