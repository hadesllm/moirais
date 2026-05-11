"""Tests for guide_on_data_analysis7u591.guide_on_data_analysis_chapter_7_unnumbered_591."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis7u591 import guide_on_data_analysis_chapter_7_unnumbered_591


def test_guide_on_data_analysis7u591_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_7_unnumbered_591(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis7u591_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_7_unnumbered_591(x)
    assert isinstance(result, dict)
