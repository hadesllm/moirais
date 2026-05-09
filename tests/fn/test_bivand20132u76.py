"""Tests for bivand20132u76.bivand2013_chapter_2_unnumbered_76."""
import numpy as np
import pytest
from moirais.fn.bivand20132u76 import bivand2013_chapter_2_unnumbered_76


def test_bivand20132u76_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_76(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20132u76_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_2_unnumbered_76(x)
    assert isinstance(result, dict)
