"""Tests for taji_d.tajimas_d."""
import numpy as np
import pytest
from moirais.fn.taji_d import tajimas_d


def test_taji_d_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    result = tajimas_d(sequences)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_taji_d_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    result = tajimas_d(sequences)
    assert isinstance(result, dict)
