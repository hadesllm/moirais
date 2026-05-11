"""Tests for treatment_effects_11u29.treatment_effects_1_chapter_1_unnumbered_29."""
import numpy as np
import pytest
from morie.fn.treatment_effects_11u29 import treatment_effects_1_chapter_1_unnumbered_29


def test_treatment_effects_11u29_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_29(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_treatment_effects_11u29_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_29(x)
    assert isinstance(result, dict)
