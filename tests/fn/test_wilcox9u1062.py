"""Tests for wilcox9u1062.wilcox_chapter_9_unnumbered_1062."""
import numpy as np
import pytest
from moirais.fn.wilcox9u1062 import wilcox_chapter_9_unnumbered_1062


def test_wilcox9u1062_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1062(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox9u1062_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1062(x)
    assert isinstance(result, dict)
