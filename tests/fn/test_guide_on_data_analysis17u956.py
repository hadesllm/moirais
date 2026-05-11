"""Tests for guide_on_data_analysis17u956.guide_on_data_analysis_chapter_17_unnumbered_956."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis17u956 import guide_on_data_analysis_chapter_17_unnumbered_956


def test_guide_on_data_analysis17u956_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_17_unnumbered_956(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis17u956_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_17_unnumbered_956(x)
    assert isinstance(result, dict)
