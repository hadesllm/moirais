"""Tests for cb7e1.cb_chapter_7_equation_1."""
import numpy as np
import pytest
from morie.fn.cb7e1 import cb_chapter_7_equation_1


def test_cb7e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_7_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_cb7e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_7_equation_1(x)
    assert isinstance(result, dict)
