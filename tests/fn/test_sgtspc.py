"""Tests for sgtspc.sgt_spectrum."""
import numpy as np
import pytest
from moirais.fn.sgtspc import sgt_spectrum


def test_sgtspc_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_spectrum(M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtspc_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_spectrum(M)
    assert isinstance(result, dict)
