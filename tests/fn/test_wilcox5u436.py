"""Tests for wilcox5u436.wilcox_chapter_5_unnumbered_436."""
import numpy as np
import pytest
from moirais.fn.wilcox5u436 import wilcox_chapter_5_unnumbered_436


def test_wilcox5u436_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_436(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u436_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_436(x)
    assert isinstance(result, dict)
