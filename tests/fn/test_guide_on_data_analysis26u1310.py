"""Tests for guide_on_data_analysis26u1310.guide_on_data_analysis_chapter_26_unnumbered_1310."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis26u1310 import guide_on_data_analysis_chapter_26_unnumbered_1310


def test_guide_on_data_analysis26u1310_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1310(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis26u1310_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_26_unnumbered_1310(x)
    assert isinstance(result, dict)
