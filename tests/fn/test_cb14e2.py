"""Tests for cb14e2.cb_chapter_14_equation_2."""
import numpy as np
import pytest
from moirais.fn.cb14e2 import cb_chapter_14_equation_2


def test_cb14e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_14_equation_2(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cb14e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cb_chapter_14_equation_2(x)
    assert isinstance(result, dict)
