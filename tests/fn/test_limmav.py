"""Tests for limmav.limma_voom."""
import numpy as np
import pytest
from morie.fn.limmav import limma_voom


def test_limmav_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = limma_voom(counts, design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_limmav_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = limma_voom(counts, design)
    assert isinstance(result, dict)
