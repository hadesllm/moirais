"""Tests for information_theory_mackay1e40.information_theory_mackay_chapter_1_equation_40."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay1e40 import information_theory_mackay_chapter_1_equation_40


def test_information_theory_mackay1e40_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_1_equation_40(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay1e40_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_1_equation_40(x)
    assert isinstance(result, dict)
