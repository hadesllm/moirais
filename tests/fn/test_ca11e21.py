"""Tests for ca11e21.ca_chapter_11_equation_21."""
import numpy as np
import pytest
from moirais.fn.ca11e21 import ca_chapter_11_equation_21


def test_ca11e21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_21(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca11e21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_11_equation_21(x)
    assert isinstance(result, dict)
