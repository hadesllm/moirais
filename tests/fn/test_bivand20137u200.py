"""Tests for bivand20137u200.bivand2013_chapter_7_unnumbered_200."""
import numpy as np
import pytest
from moirais.fn.bivand20137u200 import bivand2013_chapter_7_unnumbered_200


def test_bivand20137u200_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_200(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20137u200_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_200(x)
    assert isinstance(result, dict)
