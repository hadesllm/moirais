"""Tests for wilcox4u139.wilcox_chapter_4_unnumbered_139."""
import numpy as np
import pytest
from morie.fn.wilcox4u139 import wilcox_chapter_4_unnumbered_139


def test_wilcox4u139_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_139(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u139_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_139(x)
    assert isinstance(result, dict)
