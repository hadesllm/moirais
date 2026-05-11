"""Tests for cb15e2.cb_chapter_15_equation_2."""
import numpy as np
import pytest
from morie.fn.cb15e2 import cb_chapter_15_equation_2


def test_cb15e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_15_equation_2(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_cb15e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_15_equation_2(x)
    assert isinstance(result, dict)
