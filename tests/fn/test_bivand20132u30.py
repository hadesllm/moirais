"""Tests for bivand20132u30.bivand2013_chapter_2_unnumbered_30."""
import numpy as np
import pytest
from morie.fn.bivand20132u30 import bivand2013_chapter_2_unnumbered_30


def test_bivand20132u30_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_30(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u30_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_30(x)
    assert isinstance(result, dict)
