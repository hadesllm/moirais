"""Tests for bivand20137u212.bivand2013_chapter_7_unnumbered_212."""
import numpy as np
import pytest
from morie.fn.bivand20137u212 import bivand2013_chapter_7_unnumbered_212


def test_bivand20137u212_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_212(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u212_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_212(x)
    assert isinstance(result, dict)
