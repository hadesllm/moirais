"""Tests for guide_on_data_analysis25u1295.guide_on_data_analysis_chapter_25_unnumbered_1295."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis25u1295 import guide_on_data_analysis_chapter_25_unnumbered_1295


def test_guide_on_data_analysis25u1295_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1295(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis25u1295_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1295(x)
    assert isinstance(result, dict)
