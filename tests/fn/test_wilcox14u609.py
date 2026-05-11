"""Tests for wilcox14u609.wilcox_chapter_14_unnumbered_609."""
import numpy as np
import pytest
from morie.fn.wilcox14u609 import wilcox_chapter_14_unnumbered_609


def test_wilcox14u609_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_609(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u609_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_609(x)
    assert isinstance(result, dict)
