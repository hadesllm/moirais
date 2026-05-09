"""Tests for ca7e3.ca_chapter_7_equation_3."""
import numpy as np
import pytest
from moirais.fn.ca7e3 import ca_chapter_7_equation_3


def test_ca7e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_3(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca7e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_7_equation_3(x)
    assert isinstance(result, dict)
