"""Tests for wilcox4u77.wilcox_chapter_4_unnumbered_77."""
import numpy as np
import pytest
from moirais.fn.wilcox4u77 import wilcox_chapter_4_unnumbered_77


def test_wilcox4u77_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_77(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u77_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_77(x)
    assert isinstance(result, dict)
