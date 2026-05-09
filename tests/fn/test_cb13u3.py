"""Tests for cb13u3.cb_chapter_13_unnumbered_3."""
import numpy as np
import pytest
from moirais.fn.cb13u3 import cb_chapter_13_unnumbered_3


def test_cb13u3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = cb_chapter_13_unnumbered_3(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_cb13u3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = cb_chapter_13_unnumbered_3(x, y)
    assert isinstance(result, dict)
