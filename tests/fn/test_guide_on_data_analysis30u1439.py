"""Tests for guide_on_data_analysis30u1439.guide_on_data_analysis_chapter_30_unnumbered_1439."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis30u1439 import guide_on_data_analysis_chapter_30_unnumbered_1439


def test_guide_on_data_analysis30u1439_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1439(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_guide_on_data_analysis30u1439_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_30_unnumbered_1439(x)
    assert isinstance(result, dict)
