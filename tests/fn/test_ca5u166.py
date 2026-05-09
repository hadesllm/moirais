"""Tests for ca5u166.ca_chapter_5_unnumbered_166."""
import numpy as np
import pytest
from moirais.fn.ca5u166 import ca_chapter_5_unnumbered_166


def test_ca5u166_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_166(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca5u166_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_5_unnumbered_166(x)
    assert isinstance(result, dict)
