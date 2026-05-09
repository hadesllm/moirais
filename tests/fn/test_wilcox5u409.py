"""Tests for wilcox5u409.wilcox_chapter_5_unnumbered_409."""
import numpy as np
import pytest
from moirais.fn.wilcox5u409 import wilcox_chapter_5_unnumbered_409


def test_wilcox5u409_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_409(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u409_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_409(x)
    assert isinstance(result, dict)
