"""Tests for wilcox4u142.wilcox_chapter_4_unnumbered_142."""
import numpy as np
import pytest
from moirais.fn.wilcox4u142 import wilcox_chapter_4_unnumbered_142


def test_wilcox4u142_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_142(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u142_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_142(x)
    assert isinstance(result, dict)
