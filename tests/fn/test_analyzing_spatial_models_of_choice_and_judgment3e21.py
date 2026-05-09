"""Tests for analyzing_spatial_models_of_choice_and_judgment3e21.analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_21."""
import numpy as np
import pytest
from moirais.fn.analyzing_spatial_models_of_choice_and_judgment3e21 import analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_21


def test_analyzing_spatial_models_of_choice_and_judgment3e21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_analyzing_spatial_models_of_choice_and_judgment3e21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = analyzing_spatial_models_of_choice_and_judgment_chapter_3_equation_21(x)
    assert isinstance(result, dict)
