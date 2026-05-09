"""Tests for bivand20132u58.bivand2013_chapter_2_unnumbered_58."""
import numpy as np
import pytest
from moirais.fn.bivand20132u58 import bivand2013_chapter_2_unnumbered_58


def test_bivand20132u58_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_58(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20132u58_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_58(x)
    assert isinstance(result, dict)
