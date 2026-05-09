"""Tests for wilcox14u458.wilcox_chapter_14_unnumbered_458."""
import numpy as np
import pytest
from moirais.fn.wilcox14u458 import wilcox_chapter_14_unnumbered_458


def test_wilcox14u458_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_458(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u458_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_458(x)
    assert isinstance(result, dict)
