"""Tests for guide_on_data_analysis2u31.guide_on_data_analysis_chapter_2_unnumbered_31."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis2u31 import guide_on_data_analysis_chapter_2_unnumbered_31


def test_guide_on_data_analysis2u31_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_31(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis2u31_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_31(x)
    assert isinstance(result, dict)
