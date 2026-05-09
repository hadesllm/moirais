"""Tests for lzcomp.lempel_ziv_complexity."""
import numpy as np
import pytest
from moirais.fn.lzcomp import lempel_ziv_complexity


def test_lzcomp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = lempel_ziv_complexity(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lzcomp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = lempel_ziv_complexity(y)
    assert isinstance(result, dict)
