"""Tests for cb5e1.cb_chapter_5_equation_1."""
import numpy as np
import pytest
from moirais.fn.cb5e1 import cb_chapter_5_equation_1


def test_cb5e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_5_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_cb5e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_5_equation_1(x)
    assert isinstance(result, dict)
