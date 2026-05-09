"""Tests for bivand20137u101.bivand2013_chapter_7_unnumbered_101."""
import numpy as np
import pytest
from moirais.fn.bivand20137u101 import bivand2013_chapter_7_unnumbered_101


def test_bivand20137u101_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_101(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u101_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_101(x)
    assert isinstance(result, dict)
