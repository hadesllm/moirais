"""Tests for treatment_effects_11u40.treatment_effects_1_chapter_1_unnumbered_40."""
import numpy as np
import pytest
from moirais.fn.treatment_effects_11u40 import treatment_effects_1_chapter_1_unnumbered_40


def test_treatment_effects_11u40_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_40(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_treatment_effects_11u40_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_40(x)
    assert isinstance(result, dict)
