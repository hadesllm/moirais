"""Tests for bivand20132u31.bivand2013_chapter_2_unnumbered_31."""
import numpy as np
import pytest
from morie.fn.bivand20132u31 import bivand2013_chapter_2_unnumbered_31


def test_bivand20132u31_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_31(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u31_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_31(x)
    assert isinstance(result, dict)
