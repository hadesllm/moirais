"""Tests for information_theory_mackay19e14.information_theory_mackay_chapter_19_equation_14."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay19e14 import information_theory_mackay_chapter_19_equation_14


def test_information_theory_mackay19e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_19_equation_14(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay19e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_19_equation_14(x)
    assert isinstance(result, dict)
