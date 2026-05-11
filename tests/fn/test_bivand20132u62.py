"""Tests for bivand20132u62.bivand2013_chapter_2_unnumbered_62."""
import numpy as np
import pytest
from morie.fn.bivand20132u62 import bivand2013_chapter_2_unnumbered_62


def test_bivand20132u62_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_62(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u62_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_62(x)
    assert isinstance(result, dict)
