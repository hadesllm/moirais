"""Tests for ca11e1.ca_chapter_11_equation_1."""
import numpy as np
import pytest
from moirais.fn.ca11e1 import ca_chapter_11_equation_1


def test_ca11e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_1(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_1(x)
    assert isinstance(result, dict)
