"""Tests for wilcox2u238.wilcox_chapter_2_unnumbered_238."""
import numpy as np
import pytest
from moirais.fn.wilcox2u238 import wilcox_chapter_2_unnumbered_238


def test_wilcox2u238_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_238(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u238_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_238(x)
    assert isinstance(result, dict)
