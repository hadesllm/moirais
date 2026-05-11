"""Tests for bivand20132u7.bivand2013_chapter_2_unnumbered_7."""
import numpy as np
import pytest
from morie.fn.bivand20132u7 import bivand2013_chapter_2_unnumbered_7


def test_bivand20132u7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_7(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_7(x)
    assert isinstance(result, dict)
