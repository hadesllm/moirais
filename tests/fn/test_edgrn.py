"""Tests for edgrn.edger_diff."""
import numpy as np
import pytest
from morie.fn.edgrn import edger_diff


def test_edgrn_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = edger_diff(counts, design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_edgrn_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = edger_diff(counts, design)
    assert isinstance(result, dict)
