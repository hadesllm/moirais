"""Tests for grdino.geron_dino_self_distillation."""
import numpy as np
import pytest
from morie.fn.grdino import geron_dino_self_distillation


def test_grdino_basic():
    """Test basic functionality."""
    student_logits = np.random.default_rng(42).normal(0, 1, 100)
    teacher_logits = np.random.default_rng(42).normal(0, 1, 100)
    tau_s = np.random.default_rng(42).normal(0, 1, 100)
    tau_t = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dino_self_distillation(student_logits, teacher_logits, tau_s, tau_t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdino_edge():
    """Test edge cases."""
    student_logits = np.random.default_rng(42).normal(0, 1, 100)
    teacher_logits = np.random.default_rng(42).normal(0, 1, 100)
    tau_s = np.random.default_rng(42).normal(0, 1, 100)
    tau_t = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_dino_self_distillation(student_logits, teacher_logits, tau_s, tau_t)
    assert isinstance(result, dict)
