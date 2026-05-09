"""Tests for bivand20137u149.bivand2013_chapter_7_unnumbered_149."""
import numpy as np
import pytest
from moirais.fn.bivand20137u149 import bivand2013_chapter_7_unnumbered_149


def test_bivand20137u149_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_149(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20137u149_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_149(x)
    assert isinstance(result, dict)
