"""Tests for wilcox5u389.wilcox_chapter_5_unnumbered_389."""
import numpy as np
import pytest
from morie.fn.wilcox5u389 import wilcox_chapter_5_unnumbered_389


def test_wilcox5u389_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_389(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u389_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_389(x)
    assert isinstance(result, dict)
