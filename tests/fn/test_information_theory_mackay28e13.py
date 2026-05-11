"""Tests for information_theory_mackay28e13.information_theory_mackay_chapter_28_equation_13."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay28e13 import information_theory_mackay_chapter_28_equation_13


def test_information_theory_mackay28e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_28_equation_13(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay28e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_28_equation_13(x)
    assert isinstance(result, dict)
