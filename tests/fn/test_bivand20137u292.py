"""Tests for bivand20137u292.bivand2013_chapter_7_unnumbered_292."""
import numpy as np
import pytest
from moirais.fn.bivand20137u292 import bivand2013_chapter_7_unnumbered_292


def test_bivand20137u292_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_292(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bivand20137u292_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_7_unnumbered_292(x)
    assert isinstance(result, dict)
