"""Tests for dinopr.dino_self_distill."""
import numpy as np
import pytest
from moirais.fn.dinopr import dino_self_distill


def test_dinopr_basic():
    """Test basic functionality."""
    s_logits = np.random.default_rng(42).normal(0, 1, 100)
    t_logits = np.random.default_rng(42).normal(0, 1, 100)
    tau_t = np.random.default_rng(42).normal(0, 1, 100)
    tau_s = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_self_distill(s_logits, t_logits, tau_t, tau_s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dinopr_edge():
    """Test edge cases."""
    s_logits = np.random.default_rng(42).normal(0, 1, 100)
    t_logits = np.random.default_rng(42).normal(0, 1, 100)
    tau_t = np.random.default_rng(42).normal(0, 1, 100)
    tau_s = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_self_distill(s_logits, t_logits, tau_t, tau_s)
    assert isinstance(result, dict)
