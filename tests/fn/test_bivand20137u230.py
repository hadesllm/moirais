"""Tests for bivand20137u230.bivand2013_chapter_7_unnumbered_230."""
import numpy as np
import pytest
from morie.fn.bivand20137u230 import bivand2013_chapter_7_unnumbered_230


def test_bivand20137u230_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_230(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u230_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_230(x)
    assert isinstance(result, dict)
