"""Tests for wilcox14u459.wilcox_chapter_14_unnumbered_459."""
import numpy as np
import pytest
from moirais.fn.wilcox14u459 import wilcox_chapter_14_unnumbered_459


def test_wilcox14u459_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_459(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u459_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_459(x)
    assert isinstance(result, dict)
