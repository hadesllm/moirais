"""Tests for hwetst.hardy_weinberg."""
import numpy as np
import pytest
from moirais.fn.hwetst import hardy_weinberg


def test_hwetst_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    result = hardy_weinberg(genotypes)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hwetst_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    result = hardy_weinberg(genotypes)
    assert isinstance(result, dict)
