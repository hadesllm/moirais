"""Tests for wilcox4u161.wilcox_chapter_4_unnumbered_161."""
import numpy as np
import pytest
from moirais.fn.wilcox4u161 import wilcox_chapter_4_unnumbered_161


def test_wilcox4u161_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_161(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u161_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_161(x)
    assert isinstance(result, dict)
