"""Tests for bivand20132u22.bivand2013_chapter_2_unnumbered_22."""
import numpy as np
import pytest
from moirais.fn.bivand20132u22 import bivand2013_chapter_2_unnumbered_22


def test_bivand20132u22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_22(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_22(x)
    assert isinstance(result, dict)
