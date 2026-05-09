"""Tests for guide_on_data_analysis2u72.guide_on_data_analysis_chapter_2_unnumbered_72."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis2u72 import guide_on_data_analysis_chapter_2_unnumbered_72


def test_guide_on_data_analysis2u72_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_72(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis2u72_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_2_unnumbered_72(x)
    assert isinstance(result, dict)
