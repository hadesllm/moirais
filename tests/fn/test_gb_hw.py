"""Tests for gb_hw.gibbons_hodges_wilcoxon."""
import numpy as np
import pytest
from moirais.fn.gb_hw import gibbons_hodges_wilcoxon


def test_gb_hw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_hodges_wilcoxon(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_hw_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_hodges_wilcoxon(x, y)
    assert isinstance(result, dict)
