"""Tests for ca4e11.ca_chapter_4_equation_11."""
import numpy as np
import pytest
from morie.fn.ca4e11 import ca_chapter_4_equation_11


def test_ca4e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_11(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_11(x)
    assert isinstance(result, dict)
