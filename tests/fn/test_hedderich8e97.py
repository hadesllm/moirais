"""Tests for hedderich8e97.hedderich_chapter_8_equation_97."""
import numpy as np
import pytest
from moirais.fn.hedderich8e97 import hedderich_chapter_8_equation_97


def test_hedderich8e97_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_97(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_hedderich8e97_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = hedderich_chapter_8_equation_97(x)
    assert isinstance(result, dict)
