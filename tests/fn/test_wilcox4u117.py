"""Tests for wilcox4u117.wilcox_chapter_4_unnumbered_117."""
import numpy as np
import pytest
from moirais.fn.wilcox4u117 import wilcox_chapter_4_unnumbered_117


def test_wilcox4u117_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_117(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u117_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_117(x)
    assert isinstance(result, dict)
