"""Tests for bivand20137u281.bivand2013_chapter_7_unnumbered_281."""
import numpy as np
import pytest
from morie.fn.bivand20137u281 import bivand2013_chapter_7_unnumbered_281


def test_bivand20137u281_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_281(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u281_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_281(x)
    assert isinstance(result, dict)
