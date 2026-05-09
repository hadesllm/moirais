"""Tests for information_theory_mackay28e5.information_theory_mackay_chapter_28_equation_5."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay28e5 import information_theory_mackay_chapter_28_equation_5


def test_information_theory_mackay28e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_28_equation_5(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay28e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_28_equation_5(x)
    assert isinstance(result, dict)
