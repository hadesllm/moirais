"""Tests for sv_dl.structural_variant."""
import numpy as np
import pytest
from morie.fn.sv_dl import structural_variant


def test_sv_dl_basic():
    """Test basic functionality."""
    bam = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = structural_variant(bam, reference)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sv_dl_edge():
    """Test edge cases."""
    bam = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = structural_variant(bam, reference)
    assert isinstance(result, dict)
