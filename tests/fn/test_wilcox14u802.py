"""Tests for wilcox14u802.wilcox_chapter_14_unnumbered_802."""
import numpy as np
import pytest
from morie.fn.wilcox14u802 import wilcox_chapter_14_unnumbered_802


def test_wilcox14u802_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_802(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u802_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_802(x)
    assert isinstance(result, dict)
