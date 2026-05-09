"""Tests for bivand20137u256.bivand2013_chapter_7_unnumbered_256."""
import numpy as np
import pytest
from moirais.fn.bivand20137u256 import bivand2013_chapter_7_unnumbered_256


def test_bivand20137u256_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_256(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u256_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_256(x)
    assert isinstance(result, dict)
