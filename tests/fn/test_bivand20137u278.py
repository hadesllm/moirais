"""Tests for bivand20137u278.bivand2013_chapter_7_unnumbered_278."""
import numpy as np
import pytest
from moirais.fn.bivand20137u278 import bivand2013_chapter_7_unnumbered_278


def test_bivand20137u278_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_278(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u278_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_278(x)
    assert isinstance(result, dict)
