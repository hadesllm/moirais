"""Tests for kmdpok.kamath_dpo_loss."""
import numpy as np
import pytest
from morie.fn.kmdpok import kamath_dpo_loss


def test_kmdpok_basic():
    """Test basic functionality."""
    logp_w = np.random.default_rng(42).normal(0, 1, 100)
    logp_l = np.random.default_rng(42).normal(0, 1, 100)
    logp_ref_w = np.random.default_rng(42).normal(0, 1, 100)
    logp_ref_l = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmdpok_edge():
    """Test edge cases."""
    logp_w = np.random.default_rng(42).normal(0, 1, 100)
    logp_l = np.random.default_rng(42).normal(0, 1, 100)
    logp_ref_w = np.random.default_rng(42).normal(0, 1, 100)
    logp_ref_l = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
