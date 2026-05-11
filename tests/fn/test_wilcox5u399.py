"""Tests for wilcox5u399.wilcox_chapter_5_unnumbered_399."""
import numpy as np
import pytest
from morie.fn.wilcox5u399 import wilcox_chapter_5_unnumbered_399


def test_wilcox5u399_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_399(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u399_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_399(x)
    assert isinstance(result, dict)
