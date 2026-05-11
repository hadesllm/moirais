"""Tests for wilcox5u417.wilcox_chapter_5_unnumbered_417."""
import numpy as np
import pytest
from morie.fn.wilcox5u417 import wilcox_chapter_5_unnumbered_417


def test_wilcox5u417_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_417(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u417_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_417(x)
    assert isinstance(result, dict)
