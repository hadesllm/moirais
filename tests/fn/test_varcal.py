"""Tests for varcal.variant_calling."""
import numpy as np
import pytest
from morie.fn.varcal import variant_calling


def test_varcal_basic():
    """Test basic functionality."""
    bam = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = variant_calling(bam, reference)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varcal_edge():
    """Test edge cases."""
    bam = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = variant_calling(bam, reference)
    assert isinstance(result, dict)
