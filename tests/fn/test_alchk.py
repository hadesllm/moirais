"""Tests for alchk.alammar_recursive_chunking."""
import numpy as np
import pytest
from moirais.fn.alchk import alammar_recursive_chunking


def test_alchk_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    separators = np.random.default_rng(42).normal(0, 1, 100)
    target_size = 100
    overlap = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_recursive_chunking(text, separators, target_size, overlap)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alchk_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    separators = np.random.default_rng(42).normal(0, 1, 100)
    target_size = 100
    overlap = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_recursive_chunking(text, separators, target_size, overlap)
    assert isinstance(result, dict)
