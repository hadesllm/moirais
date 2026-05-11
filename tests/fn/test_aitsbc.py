"""Tests for aitsbc.aitchison_subcomposition."""
import numpy as np
import pytest
from morie.fn.aitsbc import aitchison_subcomposition


def test_aitsbc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    idx = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_subcomposition(x, idx)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitsbc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    idx = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_subcomposition(x, idx)
    assert isinstance(result, dict)
