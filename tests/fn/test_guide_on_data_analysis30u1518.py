"""Tests for guide_on_data_analysis30u1518.guide_on_data_analysis_chapter_30_unnumbered_1518."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis30u1518 import guide_on_data_analysis_chapter_30_unnumbered_1518


def test_guide_on_data_analysis30u1518_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1518(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis30u1518_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1518(x)
    assert isinstance(result, dict)
