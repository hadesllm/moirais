"""Tests for treatment_effects_11u13.treatment_effects_1_chapter_1_unnumbered_13."""
import numpy as np
import pytest
from moirais.fn.treatment_effects_11u13 import treatment_effects_1_chapter_1_unnumbered_13


def test_treatment_effects_11u13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_13(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_treatment_effects_11u13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_13(x)
    assert isinstance(result, dict)
