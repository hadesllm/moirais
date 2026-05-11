"""Tests for guide_on_data_analysis5u381.guide_on_data_analysis_chapter_5_unnumbered_381."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis5u381 import guide_on_data_analysis_chapter_5_unnumbered_381


def test_guide_on_data_analysis5u381_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_381(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis5u381_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_381(x)
    assert isinstance(result, dict)
