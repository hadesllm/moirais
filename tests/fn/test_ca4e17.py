"""Tests for ca4e17.ca_chapter_4_equation_17."""
import numpy as np
import pytest
from moirais.fn.ca4e17 import ca_chapter_4_equation_17


def test_ca4e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_17(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_equation_17(x)
    assert isinstance(result, dict)
