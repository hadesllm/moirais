"""Tests for guide_on_data_analysis26u1314.guide_on_data_analysis_chapter_26_unnumbered_1314."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis26u1314 import guide_on_data_analysis_chapter_26_unnumbered_1314


def test_guide_on_data_analysis26u1314_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1314(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis26u1314_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1314(x)
    assert isinstance(result, dict)
