"""Tests for naoIdx.nao_index."""
import numpy as np
import pytest
from morie.fn.naoIdx import nao_index


def test_naoIdx_basic():
    """Test basic functionality."""
    slp = np.random.default_rng(42).normal(0, 1, 100)
    result = nao_index(slp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_naoIdx_edge():
    """Test edge cases."""
    slp = np.random.default_rng(42).normal(0, 1, 100)
    result = nao_index(slp)
    assert isinstance(result, dict)
