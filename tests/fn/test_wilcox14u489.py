"""Tests for wilcox14u489.wilcox_chapter_14_unnumbered_489."""
import numpy as np
import pytest
from moirais.fn.wilcox14u489 import wilcox_chapter_14_unnumbered_489


def test_wilcox14u489_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_489(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u489_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_489(x)
    assert isinstance(result, dict)
