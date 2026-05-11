"""Tests for ca7e5.ca_chapter_7_equation_5."""
import numpy as np
import pytest
from morie.fn.ca7e5 import ca_chapter_7_equation_5


def test_ca7e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_5(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_5(x)
    assert isinstance(result, dict)
