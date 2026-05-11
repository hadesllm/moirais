"""Tests for bivand20132u2.bivand2013_chapter_2_unnumbered_2."""
import numpy as np
import pytest
from morie.fn.bivand20132u2 import bivand2013_chapter_2_unnumbered_2


def test_bivand20132u2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_2(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_2(x)
    assert isinstance(result, dict)
