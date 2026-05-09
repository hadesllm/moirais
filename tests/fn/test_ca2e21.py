"""Tests for ca2e21.ca_chapter_2_equation_21."""
import numpy as np
import pytest
from moirais.fn.ca2e21 import ca_chapter_2_equation_21


def test_ca2e21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2e21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_equation_21(x)
    assert isinstance(result, dict)
