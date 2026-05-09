"""Tests for guide_on_data_analysis30u1535.guide_on_data_analysis_chapter_30_unnumbered_1535."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis30u1535 import guide_on_data_analysis_chapter_30_unnumbered_1535


def test_guide_on_data_analysis30u1535_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1535(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis30u1535_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1535(x)
    assert isinstance(result, dict)
