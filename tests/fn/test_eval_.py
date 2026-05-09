"""Tests for moirais.fn.eval_ — E-value for unmeasured confounding."""

import math

import pytest

from moirais.fn.eval_ import e_value


def test_significant_effect_gt_one():
    """E-value should be > 1 for a significant effect (ate far from null)."""
    ev = e_value(ate=2.0, se=0.5)
    assert ev > 1.0


def test_zero_ate_returns_one():
    """E-value is 1.0 when ATE equals the null."""
    ev = e_value(ate=0.0, se=1.0, null=0.0)
    assert ev == 1.0


def test_custom_null():
    """E-value is 1.0 when ATE equals a non-zero null."""
    ev = e_value(ate=3.0, se=1.0, null=3.0)
    assert ev == 1.0


def test_negative_se_raises():
    """se <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="se must be > 0"):
        e_value(ate=1.0, se=0.0)
    with pytest.raises(ValueError, match="se must be > 0"):
        e_value(ate=1.0, se=-0.5)


def test_result_is_finite():
    """E-value should be a finite float."""
    ev = e_value(ate=1.0, se=0.3)
    assert math.isfinite(ev)
    assert isinstance(ev, float)
