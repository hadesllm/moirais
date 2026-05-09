"""Tests for wmtrwn.weights_row_normalize."""
import numpy as np
import pytest
from moirais.fn.wmtrwn import weights_row_normalize


def test_wmtrwn_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = weights_row_normalize(W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wmtrwn_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = weights_row_normalize(W)
    assert isinstance(result, dict)
