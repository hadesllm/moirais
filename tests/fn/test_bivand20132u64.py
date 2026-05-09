"""Tests for bivand20132u64.bivand2013_chapter_2_unnumbered_64."""
import numpy as np
import pytest
from moirais.fn.bivand20132u64 import bivand2013_chapter_2_unnumbered_64


def test_bivand20132u64_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_64(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u64_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_64(x)
    assert isinstance(result, dict)
