"""Tests for the 14 canonical stat callables wired in Phase 4 follow-on
(2026-05-06). These cover concepts identified in the 27,467-equation
catalog as having textbook evidence but no existing morie.fn entry.
"""

from __future__ import annotations

import numpy as np
import pytest

from morie.fn import (
    akike, bayic, covar, iqrng, ksonebs, kurt, kwallis, mad, manwhi,
    odds, relrsk, shapir, skew, wilcoxn,
)


# ── covar ───────────────────────────────────────────────────────────────────

def test_covar_known_value():
    # y = 2x → cov = 2 × var(x). var([1..5]) = 2.5 → cov = 5.0
    assert covar([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(5.0)


def test_covar_zero_corr():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    # Independent → cov ≈ 0
    assert abs(covar(a, b)) < 0.1


def test_covar_shape_mismatch():
    with pytest.raises(ValueError):
        covar([1, 2, 3], [1, 2])


# ── skew ────────────────────────────────────────────────────────────────────

def test_skew_symmetric_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    assert abs(skew(a)) < 0.2


def test_skew_positive_tail():
    # Right-skewed (one outlier) → positive
    assert skew([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]) > 1.0


# ── kurt ────────────────────────────────────────────────────────────────────

def test_kurt_uniform_negative():
    # Uniform integers 1..10 → platykurtic (excess < 0)
    assert kurt(list(range(1, 11))) < 0


def test_kurt_normal_close_to_zero():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(5000)
    assert abs(kurt(a)) < 0.3


# ── iqrng ───────────────────────────────────────────────────────────────────

def test_iqrng_basic():
    # 1..10 → Q1=3.25, Q3=7.75, IQR=4.5
    assert iqrng(list(range(1, 11))) == pytest.approx(4.5)


def test_iqrng_constant_zero():
    assert iqrng([5, 5, 5, 5, 5]) == 0


# ── mad ─────────────────────────────────────────────────────────────────────

def test_mad_normal_scaled():
    # MAD of 1..5: median=3, |xᵢ-3|=[2,1,0,1,2], median=1, ×1.4826 = 1.4826
    assert mad([1, 2, 3, 4, 5]) == pytest.approx(1.4826)


def test_mad_raw_no_scale():
    assert mad([1, 2, 3, 4, 5], scale="raw") == pytest.approx(1.0)


# ── odds ────────────────────────────────────────────────────────────────────

def test_odds_simple():
    # [[10,5],[3,7]] → (10*7) / (5*3) = 70/15 ≈ 4.667
    assert float(odds([[10, 5], [3, 7]])) == pytest.approx(70 / 15)


def test_odds_zero_cell_requires_continuity():
    with pytest.raises(ValueError):
        odds([[10, 0], [3, 7]])
    # With continuity, no error
    assert float(odds([[10, 0], [3, 7]], continuity=0.5)) > 0


# ── relrsk ──────────────────────────────────────────────────────────────────

def test_relrsk_no_effect():
    # Equal proportions in both rows → RR = 1
    # RichResult comparison ops let us compare to scalar directly.
    assert relrsk([[5, 5], [10, 10]]) == pytest.approx(1.0)


def test_relrsk_doubled():
    # Row 1 risk 10/15 ≈ 0.667; row 2 risk 3/10 = 0.3 → RR ≈ 2.222
    # relrsk now returns RichResult; float() yields scalar.
    assert float(relrsk([[10, 5], [3, 7]])) == pytest.approx(2.222, rel=1e-2)


# ── akike + bayic ───────────────────────────────────────────────────────────

def test_akike_known():
    # akike now returns RichResult; float() yields scalar AIC.
    assert float(akike(-150, 5)) == 310.0


def test_bayic_known():
    # k log(n) - 2 logL = 5 ln(100) - 2(-150) = 23.025... + 300 = 323.025
    assert float(bayic(-150, 5, 100)) == pytest.approx(323.025, abs=0.01)


def test_aic_lower_for_better_likelihood():
    assert float(akike(-100, 5)) < float(akike(-150, 5))


# ── manwhi ──────────────────────────────────────────────────────────────────

def test_manwhi_distinguishes_groups():
    out = manwhi([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert out["pvalue"] < 0.05


def test_manwhi_same_distrib_high_p():
    rng = np.random.default_rng(2)
    out = manwhi(rng.standard_normal(100).tolist(),
                 rng.standard_normal(100).tolist())
    assert out["pvalue"] > 0.05


# ── kwallis ─────────────────────────────────────────────────────────────────

def test_kwallis_distinguishes_groups():
    out = kwallis([1, 2, 3], [4, 5, 6], [7, 8, 9])
    assert out["pvalue"] < 0.1   # small samples; clear separation
    assert out["df"] == 2


def test_kwallis_one_group_raises():
    with pytest.raises(ValueError):
        kwallis([1, 2, 3])


# ── wilcoxn ─────────────────────────────────────────────────────────────────

def test_wilcoxn_paired_difference():
    # Strong systematic difference
    out = wilcoxn([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert out["pvalue"] < 0.1


def test_wilcoxn_difference_input():
    # If passing differences directly
    out = wilcoxn([1, 1, 1, 1, 1, 1, 1])
    assert "pvalue" in out


# ── ksonebs ─────────────────────────────────────────────────────────────────

def test_ksonebs_one_sample_normal():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    out = ksonebs(a, "norm")
    assert out["pvalue"] > 0.01  # reasonably consistent with Normal


def test_ksonebs_two_sample():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(200)
    b = rng.standard_normal(200) + 2.0
    out = ksonebs(a, b)
    assert out["pvalue"] < 0.001  # different means, should reject


# ── shapir ──────────────────────────────────────────────────────────────────

def test_shapir_normal_passes():
    rng = np.random.default_rng(3)
    out = shapir(rng.standard_normal(200))
    assert out["pvalue"] > 0.01


def test_shapir_uniform_rejects():
    rng = np.random.default_rng(4)
    out = shapir(rng.uniform(0, 1, size=200))
    assert out["pvalue"] < 0.05


def test_shapir_too_small():
    with pytest.raises(ValueError):
        shapir([1, 2])
