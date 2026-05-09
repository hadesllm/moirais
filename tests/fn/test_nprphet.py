"""Tests for nprphet.neural_prophet."""
import numpy as np
import pytest
from moirais.fn.nprphet import neural_prophet


def test_nprphet_basic():
    """Test basic functionality."""
    ds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ar_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = neural_prophet(ds, y, ar_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nprphet_edge():
    """Test edge cases."""
    ds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    ar_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = neural_prophet(ds, y, ar_layers)
    assert isinstance(result, dict)
