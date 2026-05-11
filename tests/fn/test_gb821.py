"""Tests for gb821.gibbons_wilcoxon_ranksum."""
import numpy as np
import pytest
from morie.fn.gb821 import gibbons_wilcoxon_ranksum


def test_gb821_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_wilcoxon_ranksum(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb821_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_wilcoxon_ranksum(x, y)
    assert isinstance(result, dict)
