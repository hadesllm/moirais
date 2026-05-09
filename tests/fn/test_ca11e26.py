"""Tests for ca11e26.ca_chapter_11_equation_26."""
import numpy as np
import pytest
from moirais.fn.ca11e26 import ca_chapter_11_equation_26


def test_ca11e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_26(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_26(x)
    assert isinstance(result, dict)
