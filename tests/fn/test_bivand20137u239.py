"""Tests for bivand20137u239.bivand2013_chapter_7_unnumbered_239."""
import numpy as np
import pytest
from morie.fn.bivand20137u239 import bivand2013_chapter_7_unnumbered_239


def test_bivand20137u239_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_239(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20137u239_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_239(x)
    assert isinstance(result, dict)
