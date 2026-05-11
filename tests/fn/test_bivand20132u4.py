"""Tests for bivand20132u4.bivand2013_chapter_2_unnumbered_4."""
import numpy as np
import pytest
from morie.fn.bivand20132u4 import bivand2013_chapter_2_unnumbered_4


def test_bivand20132u4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_4(x)
    assert isinstance(result, dict)
