"""Tests for tqlld.turboquant_lloyd_max_codebook."""
import numpy as np
import pytest
from moirais.fn.tqlld import turboquant_lloyd_max_codebook


def test_tqlld_basic():
    """Test basic functionality."""
    bits = np.random.default_rng(42).normal(0, 1, 100)
    source_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_lloyd_max_codebook(bits, source_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqlld_edge():
    """Test edge cases."""
    bits = np.random.default_rng(42).normal(0, 1, 100)
    source_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_lloyd_max_codebook(bits, source_dist)
    assert isinstance(result, dict)
