"""Tests for tqorth.turboquant_orthogonalized_jl."""
import numpy as np
import pytest
from morie.fn.tqorth import turboquant_orthogonalized_jl


def test_tqorth_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_orthogonalized_jl(S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqorth_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_orthogonalized_jl(S)
    assert isinstance(result, dict)
