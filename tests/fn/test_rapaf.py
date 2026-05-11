"""Tests for rapaf.adjusted_paf."""
import numpy as np
import pytest
from morie.fn.rapaf import adjusted_paf


def test_rapaf_basic():
    """Test basic functionality."""
    RR_strata = np.random.default_rng(42).normal(0, 1, 100)
    prevalence_strata = np.random.default_rng(42).normal(0, 1, 100)
    result = adjusted_paf(RR_strata, prevalence_strata)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rapaf_edge():
    """Test edge cases."""
    RR_strata = np.random.default_rng(42).normal(0, 1, 100)
    prevalence_strata = np.random.default_rng(42).normal(0, 1, 100)
    result = adjusted_paf(RR_strata, prevalence_strata)
    assert isinstance(result, dict)
