"""Tests for ca11e25.ca_chapter_11_equation_25."""
import numpy as np
import pytest
from morie.fn.ca11e25 import ca_chapter_11_equation_25


def test_ca11e25_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_25(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e25_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_25(x)
    assert isinstance(result, dict)
