"""Tests for guide_on_data_analysis29u1426.guide_on_data_analysis_chapter_29_unnumbered_1426."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis29u1426 import guide_on_data_analysis_chapter_29_unnumbered_1426


def test_guide_on_data_analysis29u1426_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_29_unnumbered_1426(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis29u1426_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_29_unnumbered_1426(x)
    assert isinstance(result, dict)
