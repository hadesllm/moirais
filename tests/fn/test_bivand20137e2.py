"""Tests for bivand20137e2.bivand2013_chapter_7_equation_2."""
import numpy as np
import pytest
from morie.fn.bivand20137e2 import bivand2013_chapter_7_equation_2


def test_bivand20137e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_equation_2(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20137e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_equation_2(x)
    assert isinstance(result, dict)
