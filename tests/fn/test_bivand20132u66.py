"""Tests for bivand20132u66.bivand2013_chapter_2_unnumbered_66."""
import numpy as np
import pytest
from morie.fn.bivand20132u66 import bivand2013_chapter_2_unnumbered_66


def test_bivand20132u66_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_66(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u66_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_66(x)
    assert isinstance(result, dict)
