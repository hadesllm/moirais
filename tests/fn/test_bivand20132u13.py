"""Tests for bivand20132u13.bivand2013_chapter_2_unnumbered_13."""
import numpy as np
import pytest
from moirais.fn.bivand20132u13 import bivand2013_chapter_2_unnumbered_13


def test_bivand20132u13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_13(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_13(x)
    assert isinstance(result, dict)
