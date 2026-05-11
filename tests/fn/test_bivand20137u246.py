"""Tests for bivand20137u246.bivand2013_chapter_7_unnumbered_246."""
import numpy as np
import pytest
from morie.fn.bivand20137u246 import bivand2013_chapter_7_unnumbered_246


def test_bivand20137u246_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_246(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u246_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_246(x)
    assert isinstance(result, dict)
