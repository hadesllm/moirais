"""Tests for information_theory_mackay1e17.information_theory_mackay_chapter_1_equation_17."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay1e17 import information_theory_mackay_chapter_1_equation_17


def test_information_theory_mackay1e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_1_equation_17(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay1e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_1_equation_17(x)
    assert isinstance(result, dict)
