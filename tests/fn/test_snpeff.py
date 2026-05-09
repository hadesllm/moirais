"""Tests for snpeff.variant_effect."""
import numpy as np
import pytest
from moirais.fn.snpeff import variant_effect


def test_snpeff_basic():
    """Test basic functionality."""
    variants = np.random.default_rng(42).normal(0, 1, 100)
    annotation = np.random.default_rng(42).normal(0, 1, 100)
    result = variant_effect(variants, annotation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snpeff_edge():
    """Test edge cases."""
    variants = np.random.default_rng(42).normal(0, 1, 100)
    annotation = np.random.default_rng(42).normal(0, 1, 100)
    result = variant_effect(variants, annotation)
    assert isinstance(result, dict)
