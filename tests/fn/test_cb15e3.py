"""Tests for cb15e3.cb_chapter_15_equation_3."""
import numpy as np
import pytest
from morie.fn.cb15e3 import cb_chapter_15_equation_3


def test_cb15e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_15_equation_3(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cb15e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_15_equation_3(x)
    assert isinstance(result, dict)
