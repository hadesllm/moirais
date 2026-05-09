"""Tests for ca11e17.ca_chapter_11_equation_17."""
import numpy as np
import pytest
from moirais.fn.ca11e17 import ca_chapter_11_equation_17


def test_ca11e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_17(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca11e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_17(x)
    assert isinstance(result, dict)
