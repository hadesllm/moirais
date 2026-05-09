"""Tests for rng225.rangayyan_ch4_composite_signal_in_terms_of_g."""
import numpy as np
import pytest
from moirais.fn.rng225 import rangayyan_ch4_composite_signal_in_terms_of_g


def test_rng225_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_composite_signal_in_terms_of_g(g, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rng225_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_composite_signal_in_terms_of_g(g, n)
    assert isinstance(result, dict)
