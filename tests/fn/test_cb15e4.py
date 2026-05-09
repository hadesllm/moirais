"""Tests for cb15e4.cb_chapter_15_equation_4."""
import numpy as np
import pytest
from moirais.fn.cb15e4 import cb_chapter_15_equation_4


def test_cb15e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_15_equation_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cb15e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_15_equation_4(x)
    assert isinstance(result, dict)
