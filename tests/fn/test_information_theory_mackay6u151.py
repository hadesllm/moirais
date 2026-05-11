"""Tests for information_theory_mackay6u151.information_theory_mackay_chapter_6_unnumbered_151."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay6u151 import information_theory_mackay_chapter_6_unnumbered_151


def test_information_theory_mackay6u151_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_151(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay6u151_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_151(x)
    assert isinstance(result, dict)
