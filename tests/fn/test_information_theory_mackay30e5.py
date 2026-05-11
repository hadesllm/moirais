"""Tests for information_theory_mackay30e5.information_theory_mackay_chapter_30_equation_5."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay30e5 import information_theory_mackay_chapter_30_equation_5


def test_information_theory_mackay30e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_30_equation_5(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay30e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_30_equation_5(x)
    assert isinstance(result, dict)
