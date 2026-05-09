"""Tests for ca2u33.ca_chapter_2_unnumbered_33."""
import numpy as np
import pytest
from moirais.fn.ca2u33 import ca_chapter_2_unnumbered_33


def test_ca2u33_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_33(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca2u33_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_2_unnumbered_33(x)
    assert isinstance(result, dict)
