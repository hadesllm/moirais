"""Tests for wilcox13u1266.wilcox_chapter_13_unnumbered_1266."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1266 import wilcox_chapter_13_unnumbered_1266


def test_wilcox13u1266_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1266(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1266_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1266(x)
    assert isinstance(result, dict)
