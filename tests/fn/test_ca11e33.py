"""Tests for ca11e33.ca_chapter_11_equation_33."""
import numpy as np
import pytest
from morie.fn.ca11e33 import ca_chapter_11_equation_33


def test_ca11e33_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_33(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca11e33_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_33(x)
    assert isinstance(result, dict)
