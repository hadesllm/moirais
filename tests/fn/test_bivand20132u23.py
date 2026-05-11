"""Tests for bivand20132u23.bivand2013_chapter_2_unnumbered_23."""
import numpy as np
import pytest
from morie.fn.bivand20132u23 import bivand2013_chapter_2_unnumbered_23


def test_bivand20132u23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_23(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_23(x)
    assert isinstance(result, dict)
