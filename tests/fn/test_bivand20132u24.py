"""Tests for bivand20132u24.bivand2013_chapter_2_unnumbered_24."""
import numpy as np
import pytest
from moirais.fn.bivand20132u24 import bivand2013_chapter_2_unnumbered_24


def test_bivand20132u24_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_24(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u24_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_24(x)
    assert isinstance(result, dict)
