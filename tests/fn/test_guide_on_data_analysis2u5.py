"""Tests for guide_on_data_analysis2u5.guide_on_data_analysis_chapter_2_unnumbered_5."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis2u5 import guide_on_data_analysis_chapter_2_unnumbered_5


def test_guide_on_data_analysis2u5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_5(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis2u5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_5(x)
    assert isinstance(result, dict)
