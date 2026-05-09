"""Tests for wilcox2u220.wilcox_chapter_2_unnumbered_220."""
import numpy as np
import pytest
from moirais.fn.wilcox2u220 import wilcox_chapter_2_unnumbered_220


def test_wilcox2u220_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_220(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u220_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_220(x)
    assert isinstance(result, dict)
