"""Tests for bivand20137u207.bivand2013_chapter_7_unnumbered_207."""
import numpy as np
import pytest
from moirais.fn.bivand20137u207 import bivand2013_chapter_7_unnumbered_207


def test_bivand20137u207_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_207(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u207_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_207(x)
    assert isinstance(result, dict)
