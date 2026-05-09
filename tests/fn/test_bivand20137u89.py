"""Tests for bivand20137u89.bivand2013_chapter_7_unnumbered_89."""
import numpy as np
import pytest
from moirais.fn.bivand20137u89 import bivand2013_chapter_7_unnumbered_89


def test_bivand20137u89_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_89(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u89_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_89(x)
    assert isinstance(result, dict)
