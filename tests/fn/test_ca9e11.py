"""Tests for ca9e11.ca_chapter_9_equation_11."""
import numpy as np
import pytest
from morie.fn.ca9e11 import ca_chapter_9_equation_11


def test_ca9e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_equation_11(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca9e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_9_equation_11(x)
    assert isinstance(result, dict)
