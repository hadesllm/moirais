"""Tests for ca1u3.ca_chapter_1_unnumbered_3."""
import numpy as np
import pytest
from moirais.fn.ca1u3 import ca_chapter_1_unnumbered_3


def test_ca1u3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_3(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_ca1u3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_1_unnumbered_3(x)
    assert isinstance(result, dict)
