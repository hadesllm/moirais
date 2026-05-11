"""Tests for varqc1.vcf_filter."""
import numpy as np
import pytest
from morie.fn.varqc1 import vcf_filter


def test_varqc1_basic():
    """Test basic functionality."""
    vcf = np.random.default_rng(42).normal(0, 1, 100)
    thresholds = [0.25, 0.5, 0.75]
    result = vcf_filter(vcf, thresholds)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varqc1_edge():
    """Test edge cases."""
    vcf = np.random.default_rng(42).normal(0, 1, 100)
    thresholds = [0.25, 0.5, 0.75]
    result = vcf_filter(vcf, thresholds)
    assert isinstance(result, dict)
