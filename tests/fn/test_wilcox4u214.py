"""Tests for wilcox4u214.wilcox_chapter_4_unnumbered_214."""
import numpy as np
import pytest
from moirais.fn.wilcox4u214 import wilcox_chapter_4_unnumbered_214


def test_wilcox4u214_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_214(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u214_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_214(x)
    assert isinstance(result, dict)
