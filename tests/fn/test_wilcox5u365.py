"""Tests for wilcox5u365.wilcox_chapter_5_unnumbered_365."""
import numpy as np
import pytest
from morie.fn.wilcox5u365 import wilcox_chapter_5_unnumbered_365


def test_wilcox5u365_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_365(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u365_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_365(x)
    assert isinstance(result, dict)
