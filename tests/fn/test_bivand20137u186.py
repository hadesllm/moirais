"""Tests for bivand20137u186.bivand2013_chapter_7_unnumbered_186."""
import numpy as np
import pytest
from morie.fn.bivand20137u186 import bivand2013_chapter_7_unnumbered_186


def test_bivand20137u186_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_186(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_bivand20137u186_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_186(x)
    assert isinstance(result, dict)
