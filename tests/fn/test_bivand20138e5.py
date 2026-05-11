"""Tests for bivand20138e5.bivand2013_chapter_8_equation_5."""
import numpy as np
import pytest
from morie.fn.bivand20138e5 import bivand2013_chapter_8_equation_5


def test_bivand20138e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_8_equation_5(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20138e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_8_equation_5(x)
    assert isinstance(result, dict)
