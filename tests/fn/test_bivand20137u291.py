"""Tests for bivand20137u291.bivand2013_chapter_7_unnumbered_291."""
import numpy as np
import pytest
from morie.fn.bivand20137u291 import bivand2013_chapter_7_unnumbered_291


def test_bivand20137u291_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_291(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20137u291_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_291(x)
    assert isinstance(result, dict)
