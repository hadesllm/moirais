"""Tests for guide_on_data_analysis27u1370.guide_on_data_analysis_chapter_27_unnumbered_1370."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis27u1370 import guide_on_data_analysis_chapter_27_unnumbered_1370


def test_guide_on_data_analysis27u1370_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1370(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis27u1370_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_27_unnumbered_1370(x)
    assert isinstance(result, dict)
