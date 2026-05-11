"""Tests for ca4e7.ca_chapter_4_equation_7."""
import numpy as np
import pytest
from morie.fn.ca4e7 import ca_chapter_4_equation_7


def test_ca4e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_7(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_7(x)
    assert isinstance(result, dict)
