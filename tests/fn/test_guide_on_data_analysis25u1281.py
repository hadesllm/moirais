"""Tests for guide_on_data_analysis25u1281.guide_on_data_analysis_chapter_25_unnumbered_1281."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis25u1281 import guide_on_data_analysis_chapter_25_unnumbered_1281


def test_guide_on_data_analysis25u1281_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1281(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis25u1281_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_25_unnumbered_1281(x)
    assert isinstance(result, dict)
