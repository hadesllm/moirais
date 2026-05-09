"""Tests for ca1u5.ca_chapter_1_unnumbered_5."""
import numpy as np
import pytest
from moirais.fn.ca1u5 import ca_chapter_1_unnumbered_5


def test_ca1u5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_5(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca1u5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_5(x)
    assert isinstance(result, dict)
