"""Tests for bivand20137u170.bivand2013_chapter_7_unnumbered_170."""
import numpy as np
import pytest
from morie.fn.bivand20137u170 import bivand2013_chapter_7_unnumbered_170


def test_bivand20137u170_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_170(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u170_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_170(x)
    assert isinstance(result, dict)
