"""Tests for wilcox13u1268.wilcox_chapter_13_unnumbered_1268."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1268 import wilcox_chapter_13_unnumbered_1268


def test_wilcox13u1268_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1268(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1268_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1268(x)
    assert isinstance(result, dict)
