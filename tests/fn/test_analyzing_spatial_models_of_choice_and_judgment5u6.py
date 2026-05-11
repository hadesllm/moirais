"""Tests for analyzing_spatial_models_of_choice_and_judgment5u6.analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_6."""
import numpy as np
import pytest
from morie.fn.analyzing_spatial_models_of_choice_and_judgment5u6 import analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_6


def test_analyzing_spatial_models_of_choice_and_judgment5u6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analyzing_spatial_models_of_choice_and_judgment5u6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_6(x)
    assert isinstance(result, dict)
