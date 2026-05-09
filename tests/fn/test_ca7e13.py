"""Tests for ca7e13.ca_chapter_7_equation_13."""
import numpy as np
import pytest
from moirais.fn.ca7e13 import ca_chapter_7_equation_13


def test_ca7e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_13(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_13(x)
    assert isinstance(result, dict)
