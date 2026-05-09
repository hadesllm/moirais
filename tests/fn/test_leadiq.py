"""Tests for Lanphear 2005 Lead → IQ loss model."""

import math

import numpy as np
import pytest

from moirais.fn.leadiq import leadiq, lead_iq_loss


def test_leadiq_zero_loss_at_reference():
    r = leadiq(1.0, reference_bll=1.0)
    assert r.value == pytest.approx(0.0)


def test_leadiq_reproduces_lanphear_1_to_10_contrast():
    # Paper: BLL 1 -> 10 µg/dL estimated at -6.2 IQ points.
    # Our model with a=-2.7: ΔIQ = -2.7 * ln(10) = -6.217
    r = leadiq(10.0, reference_bll=1.0)
    assert r.value == pytest.approx(-6.217, abs=0.01)


def test_leadiq_is_sublinear_per_unit_at_higher_bll():
    # Effect per unit BLL is LARGER at lower concentrations (key finding).
    # d(IQ)/d(BLL) = a / BLL; compare derivative at BLL=2 vs BLL=20.
    r_low = leadiq(2.0)
    r_mid = leadiq(20.0)
    # Loss at 2 vs 1: a*ln(2) ≈ -1.87
    # Loss at 20 vs 1: a*ln(20) ≈ -8.09
    # So 1→2 costs 1.87, 10→20 costs 1.87 (same ratio). Log-linearity
    # means EQUAL fractional steps give EQUAL loss. But the UNIT
    # (per-µg/dL) effect IS larger at low BLL:
    # (1.87 / 1 unit) at low vs (1.87 / 10 units) at high.
    loss_per_unit_low = abs(leadiq(2.0).value) / 1.0
    loss_per_unit_high = (abs(leadiq(20.0).value)
                           - abs(leadiq(10.0).value)) / 10.0
    assert loss_per_unit_low > loss_per_unit_high


def test_leadiq_with_ci_gives_bounds():
    r = leadiq(10.0, with_ci=True)
    # Central -6.2; lower (less IQ loss) and upper (more) should bracket
    assert r.extra["iq_loss_lower_95"] > r.value   # less negative
    assert r.extra["iq_loss_upper_95"] < r.value   # more negative


def test_leadiq_array_input():
    bll = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    r = leadiq(bll)
    losses = r.extra["iq_loss"]
    assert losses[0] == pytest.approx(0.0, abs=1e-6)
    # Monotone decreasing loss as BLL increases
    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1]


def test_leadiq_rejects_nonpositive_bll():
    with pytest.raises(ValueError, match="BLL values must be > 0"):
        leadiq(0.0)
    with pytest.raises(ValueError, match="BLL values must be > 0"):
        leadiq(-1.0)


def test_leadiq_rejects_nonpositive_reference():
    with pytest.raises(ValueError, match="reference_bll must be > 0"):
        leadiq(5.0, reference_bll=0.0)


def test_leadiq_alias_matches():
    assert leadiq is lead_iq_loss
