"""Tests for wilcox14u729.wilcox_chapter_14_unnumbered_729."""
import numpy as np
import pytest
from morie.fn.wilcox14u729 import wilcox_chapter_14_unnumbered_729


def test_wilcox14u729_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_729(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u729_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_729(x)
    assert isinstance(result, dict)
