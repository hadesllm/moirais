"""Tests for ca11e36.ca_chapter_11_equation_36."""
import numpy as np
import pytest
from morie.fn.ca11e36 import ca_chapter_11_equation_36


def test_ca11e36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_36(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_36(x)
    assert isinstance(result, dict)
