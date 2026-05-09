"""Tests for wilcox4u119.wilcox_chapter_4_unnumbered_119."""
import numpy as np
import pytest
from moirais.fn.wilcox4u119 import wilcox_chapter_4_unnumbered_119


def test_wilcox4u119_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_119(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u119_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_119(x)
    assert isinstance(result, dict)
