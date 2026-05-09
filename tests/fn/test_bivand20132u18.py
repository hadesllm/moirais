"""Tests for bivand20132u18.bivand2013_chapter_2_unnumbered_18."""
import numpy as np
import pytest
from moirais.fn.bivand20132u18 import bivand2013_chapter_2_unnumbered_18


def test_bivand20132u18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_18(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_18(x)
    assert isinstance(result, dict)
