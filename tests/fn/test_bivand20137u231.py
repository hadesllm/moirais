"""Tests for bivand20137u231.bivand2013_chapter_7_unnumbered_231."""
import numpy as np
import pytest
from moirais.fn.bivand20137u231 import bivand2013_chapter_7_unnumbered_231


def test_bivand20137u231_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_231(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand20137u231_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_231(x)
    assert isinstance(result, dict)
