"""Tests for guide_on_data_analysis25u1283.guide_on_data_analysis_chapter_25_unnumbered_1283."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis25u1283 import guide_on_data_analysis_chapter_25_unnumbered_1283


def test_guide_on_data_analysis25u1283_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1283(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis25u1283_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1283(x)
    assert isinstance(result, dict)
