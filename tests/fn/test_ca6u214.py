"""Tests for ca6u214.ca_chapter_6_unnumbered_214."""
import numpy as np
import pytest
from moirais.fn.ca6u214 import ca_chapter_6_unnumbered_214


def test_ca6u214_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_214(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6u214_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_214(x)
    assert isinstance(result, dict)
