"""Tests for wilcox5u394.wilcox_chapter_5_unnumbered_394."""
import numpy as np
import pytest
from morie.fn.wilcox5u394 import wilcox_chapter_5_unnumbered_394


def test_wilcox5u394_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_394(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u394_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_394(x)
    assert isinstance(result, dict)
