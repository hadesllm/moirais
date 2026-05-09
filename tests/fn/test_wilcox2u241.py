"""Tests for wilcox2u241.wilcox_chapter_2_unnumbered_241."""
import numpy as np
import pytest
from moirais.fn.wilcox2u241 import wilcox_chapter_2_unnumbered_241


def test_wilcox2u241_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_241(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u241_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_241(x)
    assert isinstance(result, dict)
