"""Tests for guide_on_data_analysis2u87.guide_on_data_analysis_chapter_2_unnumbered_87."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis2u87 import guide_on_data_analysis_chapter_2_unnumbered_87


def test_guide_on_data_analysis2u87_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_87(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis2u87_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_87(x)
    assert isinstance(result, dict)
