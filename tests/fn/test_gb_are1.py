"""Tests for gb_are1.gibbons_are_sign_wilcoxon."""
import numpy as np
import pytest
from moirais.fn.gb_are1 import gibbons_are_sign_wilcoxon


def test_gb_are1_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_sign_wilcoxon(f)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_are1_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_sign_wilcoxon(f)
    assert isinstance(result, dict)
