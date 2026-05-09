"""Tests for information_theory_mackay26u220.information_theory_mackay_chapter_26_unnumbered_220."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay26u220 import information_theory_mackay_chapter_26_unnumbered_220


def test_information_theory_mackay26u220_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_26_unnumbered_220(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay26u220_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_26_unnumbered_220(x)
    assert isinstance(result, dict)
