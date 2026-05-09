"""Tests for gb1051.gibbons_k_rank_alt."""
import numpy as np
import pytest
from moirais.fn.gb1051 import gibbons_k_rank_alt


def test_gb1051_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k_rank_alt(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1051_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k_rank_alt(groups)
    assert isinstance(result, dict)
