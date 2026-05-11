"""Tests for the 8 crimstat textbook callables wired in Phase 1c.

These cover the canonical formulas from the two priority criminology
stats books (Wooditch 2021, Weisburd et al 2022).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from morie.fn import cohend, gkgam, kentau, mcfadr, s2var, somerd, spearm, xbar


# ── xbar ────────────────────────────────────────────────────────────────────

def test_xbar_basic():
    assert xbar([1, 2, 3, 4, 5]) == 3.0


def test_xbar_negatives():
    assert xbar([-2, 0, 2]) == 0.0


def test_xbar_empty_raises():
    with pytest.raises(ValueError):
        xbar([])


# ── s2var ───────────────────────────────────────────────────────────────────

def test_s2var_unbiased():
    # 1..5: mean=3, ssq=10, /(n-1)=4 → 2.5
    assert s2var([1, 2, 3, 4, 5]) == pytest.approx(2.5)


def test_s2var_singleton_raises():
    with pytest.raises(ValueError):
        s2var([42])


# ── cohend ──────────────────────────────────────────────────────────────────

def test_cohend_known_value():
    # Two equal-variance groups, mean diff 1, sd 1 → d ≈ 1
    # cohend now returns RichResult; float(...) gives the scalar.
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, scale=1.0, size=2000)
    b = rng.normal(loc=0.0, scale=1.0, size=2000)
    d = float(cohend(a, b))
    assert d == pytest.approx(1.0, abs=0.1)


def test_cohend_zero_diff():
    a = [1, 2, 3, 4, 5]
    assert float(cohend(a, a)) == pytest.approx(0.0)


def test_cohend_zero_pooled_sd_raises():
    with pytest.raises(ValueError):
        cohend([5, 5, 5], [5, 5, 5])


# ── mcfadr ──────────────────────────────────────────────────────────────────

def test_mcfadr_basic():
    # Full LL = -150, null LL = -200 → 1 - 150/200 = 0.25
    # mcfadr now returns RichResult; float(...) gives the scalar R².
    assert float(mcfadr(-150.0, -200.0)) == pytest.approx(0.25)


def test_mcfadr_perfect_fit():
    # Full LL = 0 (perfect prediction) → R² = 1
    assert float(mcfadr(0.0, -100.0)) == pytest.approx(1.0)


def test_mcfadr_null_zero_raises():
    with pytest.raises(ValueError):
        mcfadr(-50.0, 0.0)


# ── somerd ──────────────────────────────────────────────────────────────────

def test_somerd_perfect_concordance():
    out = somerd([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert out["statistic"] == pytest.approx(1.0)


def test_somerd_perfect_discordance():
    out = somerd([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert out["statistic"] == pytest.approx(-1.0)


# ── kentau ──────────────────────────────────────────────────────────────────

def test_kentau_perfect_concordance():
    out = kentau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert out["statistic"] == pytest.approx(1.0)


def test_kentau_perfect_discordance():
    out = kentau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert out["statistic"] == pytest.approx(-1.0)


# ── spearm ──────────────────────────────────────────────────────────────────

def test_spearm_monotonic_nonlinear():
    # Strictly monotonic but nonlinear → ρ = 1
    x = list(range(10))
    y = [v ** 2 for v in x]
    out = spearm(x, y)
    assert out["statistic"] == pytest.approx(1.0)


def test_spearm_negative_correlation():
    out = spearm([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
    assert out["statistic"] == pytest.approx(-1.0)


# ── gkgam ───────────────────────────────────────────────────────────────────

def test_gkgam_perfect_concordance():
    # gkgam now returns RichResult; float() yields scalar gamma.
    assert float(gkgam([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])) == pytest.approx(1.0)


def test_gkgam_perfect_discordance():
    assert float(gkgam([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])) == pytest.approx(-1.0)


def test_gkgam_all_ties_raises():
    with pytest.raises(ValueError):
        gkgam([1, 1, 1], [2, 2, 2])
