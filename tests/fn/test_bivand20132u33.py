"""Tests for bivand20132u33.bivand2013_chapter_2_unnumbered_33."""
import numpy as np
import pytest
from morie.fn.bivand20132u33 import bivand2013_chapter_2_unnumbered_33


def test_bivand20132u33_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_33(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u33_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_33(x)
    assert isinstance(result, dict)
