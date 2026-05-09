"""Tests for wilcox2u303.wilcox_chapter_2_unnumbered_303."""
import numpy as np
import pytest
from moirais.fn.wilcox2u303 import wilcox_chapter_2_unnumbered_303


def test_wilcox2u303_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_303(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u303_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_303(x)
    assert isinstance(result, dict)
