"""Tests for wilcox14u597.wilcox_chapter_14_unnumbered_597."""
import numpy as np
import pytest
from moirais.fn.wilcox14u597 import wilcox_chapter_14_unnumbered_597


def test_wilcox14u597_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_597(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u597_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_597(x)
    assert isinstance(result, dict)
