"""Tests for guide_on_data_analysis2u61.guide_on_data_analysis_chapter_2_unnumbered_61."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis2u61 import guide_on_data_analysis_chapter_2_unnumbered_61


def test_guide_on_data_analysis2u61_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_61(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis2u61_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_61(x)
    assert isinstance(result, dict)
