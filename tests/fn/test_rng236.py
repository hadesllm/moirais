"""Tests for rng236.rangayyan_ch4_complex_cepstrum_definition."""
import numpy as np
import pytest
from moirais.fn.rng236 import rangayyan_ch4_complex_cepstrum_definition


def test_rng236_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstrum_definition(Y, z, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng236_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_complex_cepstrum_definition(Y, z, n)
    assert isinstance(result, dict)
