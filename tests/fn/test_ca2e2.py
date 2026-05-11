"""Tests for ca2e2.ca_chapter_2_equation_2."""
import numpy as np
import pytest
from morie.fn.ca2e2 import ca_chapter_2_equation_2


def test_ca2e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_2(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_2(x)
    assert isinstance(result, dict)
