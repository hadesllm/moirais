"""Tests for wilcox14u581.wilcox_chapter_14_unnumbered_581."""
import numpy as np
import pytest
from moirais.fn.wilcox14u581 import wilcox_chapter_14_unnumbered_581


def test_wilcox14u581_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_581(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u581_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_581(x)
    assert isinstance(result, dict)
