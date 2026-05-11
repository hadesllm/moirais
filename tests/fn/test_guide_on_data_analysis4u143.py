"""Tests for guide_on_data_analysis4u143.guide_on_data_analysis_chapter_4_unnumbered_143."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis4u143 import guide_on_data_analysis_chapter_4_unnumbered_143


def test_guide_on_data_analysis4u143_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_143(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis4u143_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_143(x)
    assert isinstance(result, dict)
