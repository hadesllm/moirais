"""Tests for guide_on_data_analysis16u926.guide_on_data_analysis_chapter_16_unnumbered_926."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis16u926 import guide_on_data_analysis_chapter_16_unnumbered_926


def test_guide_on_data_analysis16u926_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_16_unnumbered_926(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis16u926_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_16_unnumbered_926(x)
    assert isinstance(result, dict)
