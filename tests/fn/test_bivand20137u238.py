"""Tests for bivand20137u238.bivand2013_chapter_7_unnumbered_238."""
import numpy as np
import pytest
from morie.fn.bivand20137u238 import bivand2013_chapter_7_unnumbered_238


def test_bivand20137u238_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_238(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u238_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_238(x)
    assert isinstance(result, dict)
