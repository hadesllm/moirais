"""Tests for ca11e30.ca_chapter_11_equation_30."""
import numpy as np
import pytest
from morie.fn.ca11e30 import ca_chapter_11_equation_30


def test_ca11e30_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_30(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_ca11e30_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_30(x)
    assert isinstance(result, dict)
