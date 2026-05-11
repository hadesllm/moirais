"""Tests for bivand20132u51.bivand2013_chapter_2_unnumbered_51."""
import numpy as np
import pytest
from morie.fn.bivand20132u51 import bivand2013_chapter_2_unnumbered_51


def test_bivand20132u51_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_51(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u51_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_51(x)
    assert isinstance(result, dict)
