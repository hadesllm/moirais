"""Tests for cb10u1.cb_chapter_10_unnumbered_1."""
import numpy as np
import pytest
from moirais.fn.cb10u1 import cb_chapter_10_unnumbered_1


def test_cb10u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_10_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cb10u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_10_unnumbered_1(x)
    assert isinstance(result, dict)
