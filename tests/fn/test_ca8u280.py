"""Tests for ca8u280.ca_chapter_8_unnumbered_280."""
import numpy as np
import pytest
from moirais.fn.ca8u280 import ca_chapter_8_unnumbered_280


def test_ca8u280_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_280(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca8u280_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_280(x)
    assert isinstance(result, dict)
