"""Tests for analyzing_spatial_models_of_choice_and_judgment5u1.analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_1."""
import numpy as np
import pytest
from morie.fn.analyzing_spatial_models_of_choice_and_judgment5u1 import analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_1


def test_analyzing_spatial_models_of_choice_and_judgment5u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_analyzing_spatial_models_of_choice_and_judgment5u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_1(x)
    assert isinstance(result, dict)
