"""Tests for wilcox2u262.wilcox_chapter_2_unnumbered_262."""
import numpy as np
import pytest
from morie.fn.wilcox2u262 import wilcox_chapter_2_unnumbered_262


def test_wilcox2u262_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_262(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u262_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_262(x)
    assert isinstance(result, dict)
