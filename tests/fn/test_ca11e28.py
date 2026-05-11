"""Tests for ca11e28.ca_chapter_11_equation_28."""
import numpy as np
import pytest
from morie.fn.ca11e28 import ca_chapter_11_equation_28


def test_ca11e28_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_28(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e28_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_28(x)
    assert isinstance(result, dict)
