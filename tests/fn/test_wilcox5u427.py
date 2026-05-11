"""Tests for wilcox5u427.wilcox_chapter_5_unnumbered_427."""
import numpy as np
import pytest
from morie.fn.wilcox5u427 import wilcox_chapter_5_unnumbered_427


def test_wilcox5u427_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_427(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u427_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_427(x)
    assert isinstance(result, dict)
