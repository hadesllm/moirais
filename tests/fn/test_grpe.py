"""Tests for grpe.geron_sinusoidal_positional_encoding."""
import numpy as np
import pytest
from moirais.fn.grpe import geron_sinusoidal_positional_encoding


def test_grpe_basic():
    """Test basic functionality."""
    seq_len = 100
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sinusoidal_positional_encoding(seq_len, d_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grpe_edge():
    """Test edge cases."""
    seq_len = 100
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sinusoidal_positional_encoding(seq_len, d_model)
    assert isinstance(result, dict)
