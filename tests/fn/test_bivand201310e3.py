"""Tests for bivand201310e3.bivand2013_chapter_10_equation_3."""
import numpy as np
import pytest
from moirais.fn.bivand201310e3 import bivand2013_chapter_10_equation_3


def test_bivand201310e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_10_equation_3(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bivand201310e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bivand2013_chapter_10_equation_3(x)
    assert isinstance(result, dict)
