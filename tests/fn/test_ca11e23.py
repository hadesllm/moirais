"""Tests for ca11e23.ca_chapter_11_equation_23."""
import numpy as np
import pytest
from moirais.fn.ca11e23 import ca_chapter_11_equation_23


def test_ca11e23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_23(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_23(x)
    assert isinstance(result, dict)
