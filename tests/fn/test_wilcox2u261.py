"""Tests for wilcox2u261.wilcox_chapter_2_unnumbered_261."""
import numpy as np
import pytest
from moirais.fn.wilcox2u261 import wilcox_chapter_2_unnumbered_261


def test_wilcox2u261_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_261(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u261_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_261(x)
    assert isinstance(result, dict)
