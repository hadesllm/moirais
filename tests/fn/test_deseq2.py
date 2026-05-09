"""Tests for deseq2.deseq2_differential."""
import numpy as np
import pytest
from moirais.fn.deseq2 import deseq2_differential


def test_deseq2_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = deseq2_differential(counts, design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deseq2_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = deseq2_differential(counts, design)
    assert isinstance(result, dict)
