"""Tests for ca11e31.ca_chapter_11_equation_31."""
import numpy as np
import pytest
from morie.fn.ca11e31 import ca_chapter_11_equation_31


def test_ca11e31_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_31(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_ca11e31_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_31(x)
    assert isinstance(result, dict)
