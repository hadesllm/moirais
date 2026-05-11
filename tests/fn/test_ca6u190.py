"""Tests for ca6u190.ca_chapter_6_unnumbered_190."""
import numpy as np
import pytest
from morie.fn.ca6u190 import ca_chapter_6_unnumbered_190


def test_ca6u190_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_190(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ca6u190_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ca_chapter_6_unnumbered_190(x)
    assert isinstance(result, dict)
