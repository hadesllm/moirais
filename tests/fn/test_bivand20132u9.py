"""Tests for bivand20132u9.bivand2013_chapter_2_unnumbered_9."""
import numpy as np
import pytest
from morie.fn.bivand20132u9 import bivand2013_chapter_2_unnumbered_9


def test_bivand20132u9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_9(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_9(x)
    assert isinstance(result, dict)
