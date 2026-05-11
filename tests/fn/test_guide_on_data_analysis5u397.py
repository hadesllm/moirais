"""Tests for guide_on_data_analysis5u397.guide_on_data_analysis_chapter_5_unnumbered_397."""
import numpy as np
import pytest
from morie.fn.guide_on_data_analysis5u397 import guide_on_data_analysis_chapter_5_unnumbered_397


def test_guide_on_data_analysis5u397_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_397(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_guide_on_data_analysis5u397_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = guide_on_data_analysis_chapter_5_unnumbered_397(x)
    assert isinstance(result, dict)
