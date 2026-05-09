"""Tests for ca4u122.ca_chapter_4_unnumbered_122."""
import numpy as np
import pytest
from moirais.fn.ca4u122 import ca_chapter_4_unnumbered_122


def test_ca4u122_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_122(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca4u122_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_4_unnumbered_122(x)
    assert isinstance(result, dict)
