"""Tests for guide_on_data_analysis2u28.guide_on_data_analysis_chapter_2_unnumbered_28."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis2u28 import guide_on_data_analysis_chapter_2_unnumbered_28


def test_guide_on_data_analysis2u28_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_28(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis2u28_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_28(x)
    assert isinstance(result, dict)
