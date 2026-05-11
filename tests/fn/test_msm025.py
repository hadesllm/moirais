"""Tests for msm025.mvsml_linear_mixed_models_eq_5_3."""
import numpy as np
import pytest
from morie.fn.msm025 import mvsml_linear_mixed_models_eq_5_3


def test_msm025_basic():
    """Test basic functionality."""
    ti = np.random.default_rng(42).normal(0, 1, 100)
    trait = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Linear = np.random.default_rng(42).normal(0, 1, 100)
    Mixed = np.random.default_rng(42).normal(0, 1, 100)
    Effects = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(ti, trait, Genomic, Linear, Mixed, Effects)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm025_edge():
    """Test edge cases."""
    ti = np.random.default_rng(42).normal(0, 1, 100)
    trait = np.random.default_rng(42).normal(0, 1, 100)
    Genomic = np.random.default_rng(42).normal(0, 1, 100)
    Linear = np.random.default_rng(42).normal(0, 1, 100)
    Mixed = np.random.default_rng(42).normal(0, 1, 100)
    Effects = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_3(ti, trait, Genomic, Linear, Mixed, Effects)
    assert isinstance(result, dict)
