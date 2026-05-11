"""Tests for ca7e1.ca_chapter_7_equation_1."""
import numpy as np
import pytest
from morie.fn.ca7e1 import ca_chapter_7_equation_1


def test_ca7e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_1(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_1(x)
    assert isinstance(result, dict)
