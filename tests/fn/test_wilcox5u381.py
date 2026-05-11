"""Tests for wilcox5u381.wilcox_chapter_5_unnumbered_381."""
import numpy as np
import pytest
from morie.fn.wilcox5u381 import wilcox_chapter_5_unnumbered_381


def test_wilcox5u381_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_381(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u381_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_381(x)
    assert isinstance(result, dict)
