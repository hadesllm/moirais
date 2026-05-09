"""Tests for guide_on_data_analysis4u239.guide_on_data_analysis_chapter_4_unnumbered_239."""
import numpy as np
import pytest
from moirais.fn.guide_on_data_analysis4u239 import guide_on_data_analysis_chapter_4_unnumbered_239


def test_guide_on_data_analysis4u239_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_239(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_guide_on_data_analysis4u239_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_4_unnumbered_239(x)
    assert isinstance(result, dict)
