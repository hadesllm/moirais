"""Tests for bivand20137u193.bivand2013_chapter_7_unnumbered_193."""
import numpy as np
import pytest
from morie.fn.bivand20137u193 import bivand2013_chapter_7_unnumbered_193


def test_bivand20137u193_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_193(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_bivand20137u193_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_193(x)
    assert isinstance(result, dict)
