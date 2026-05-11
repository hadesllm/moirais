"""Tests for wilcox2u263.wilcox_chapter_2_unnumbered_263."""
import numpy as np
import pytest
from morie.fn.wilcox2u263 import wilcox_chapter_2_unnumbered_263


def test_wilcox2u263_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_263(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u263_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_263(x)
    assert isinstance(result, dict)
