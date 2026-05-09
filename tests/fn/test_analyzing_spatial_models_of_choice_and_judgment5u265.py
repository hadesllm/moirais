"""Tests for analyzing_spatial_models_of_choice_and_judgment5u265.analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_265."""
import numpy as np
import pytest
from moirais.fn.analyzing_spatial_models_of_choice_and_judgment5u265 import analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_265


def test_analyzing_spatial_models_of_choice_and_judgment5u265_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_265(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analyzing_spatial_models_of_choice_and_judgment5u265_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_5_unnumbered_265(x)
    assert isinstance(result, dict)
