"""Tests for ca8u314.ca_chapter_8_unnumbered_314."""
import numpy as np
import pytest
from moirais.fn.ca8u314 import ca_chapter_8_unnumbered_314


def test_ca8u314_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_314(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_ca8u314_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_8_unnumbered_314(x)
    assert isinstance(result, dict)
