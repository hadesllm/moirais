"""Tests for pm_gemm_burden — PAF + attributable deaths via GEMM."""

import numpy as np
import pytest

from moirais.fn.pmgbrd import pm_gemm_burden, pmgbrd
from moirais.fn.pmgemm import pm_gemm_rr


def test_pmgbrd_at_tmrel_zero_deaths():
    # At TMREL (2.4 µg/m³), RR=1, PAF=0, attributable deaths=0.
    r = pmgbrd(2.4, 100_000, 0.008)
    assert r.value == pytest.approx(0.0)
    assert r.extra["paf"] == pytest.approx(0.0)


def test_pmgbrd_below_tmrel_zero_deaths():
    r = pmgbrd(1.0, 100_000, 0.008)
    assert r.value == pytest.approx(0.0)


def test_pmgbrd_matches_gemm_rr_at_single_point():
    # PAF = (RR - 1) / RR; deaths = PAF * baseline * pop.
    # Verify all three pieces agree with independent pmgemm call.
    conc = 25.0
    pop  = 50_000
    rate = 0.008
    r = pmgbrd(conc, pop, rate, outcome="ncd_lri")
    expected_rr = pm_gemm_rr(conc, outcome="ncd_lri").value
    expected_paf = (expected_rr - 1) / expected_rr
    expected_deaths = expected_paf * rate * pop
    assert r.extra["rr"] == pytest.approx(expected_rr, rel=1e-6)
    assert r.extra["paf"] == pytest.approx(expected_paf, rel=1e-6)
    assert r.value == pytest.approx(expected_deaths, rel=1e-6)


def test_pmgbrd_linear_in_population():
    # Doubling population should double attributable deaths.
    r1 = pmgbrd(20.0, 50_000, 0.008)
    r2 = pmgbrd(20.0, 100_000, 0.008)
    assert r2.value == pytest.approx(2 * r1.value, rel=1e-6)


def test_pmgbrd_linear_in_baseline_rate():
    r1 = pmgbrd(20.0, 100_000, 0.006)
    r2 = pmgbrd(20.0, 100_000, 0.012)
    assert r2.value == pytest.approx(2 * r1.value, rel=1e-6)


def test_pmgbrd_array_per_unit_deaths():
    # Per-FSA loop: 3 units with different C and pop.
    concs = np.array([8.0, 12.0, 15.0])
    pops  = np.array([25_000, 30_000, 20_000])
    r = pmgbrd(concs, pops, 0.008, outcome="ncd_lri")
    dpu = r.extra["deaths_per_unit"]
    assert len(dpu) == 3
    # Totals are summed
    assert r.value == pytest.approx(sum(dpu), rel=1e-9)
    # Monotone: RR rises with C, so deaths-per-unit should rise when
    # holding pop constant. Normalize by pop:
    rates = [dpu[i] / pops[i] for i in range(3)]
    assert rates[0] < rates[1] < rates[2]


def test_pmgbrd_population_broadcast_scalar():
    # Scalar population broadcasts to an array of concentrations.
    concs = np.array([5.0, 10.0, 20.0])
    r = pmgbrd(concs, 10_000, 0.008)
    assert len(r.extra["deaths_per_unit"]) == 3
    # All three see the same 10k pop.
    assert r.extra["total_population"] == 30_000


def test_pmgbrd_saturation_above_100():
    # GEMM saturates; PAF should too. PAF at 300 shouldn't be >2×
    # PAF at 100 (it'd be much more for a linear model).
    r_100 = pmgbrd(100.0, 1_000_000, 0.008).extra["paf"]
    r_300 = pmgbrd(300.0, 1_000_000, 0.008).extra["paf"]
    assert r_300 < 2 * r_100, (
        f"PAF not saturating: r_100={r_100:.3f} r_300={r_300:.3f}"
    )


def test_pmgbrd_outcome_switch_changes_answer():
    # lung_cancer has steeper θ than ncd_lri, so higher PAF at
    # moderate C (≳ 20 µg/m³ where GEMM LC slope exceeds NCD+LRI).
    c = 60.0
    r_ncd = pmgbrd(c, 100_000, 0.008, outcome="ncd_lri")
    r_lc  = pmgbrd(c, 100_000, 0.008, outcome="lung_cancer")
    assert r_lc.value > r_ncd.value


def test_pmgbrd_rate_negative_raises():
    with pytest.raises(ValueError, match="baseline_rate"):
        pmgbrd(10.0, 100_000, -0.001)


def test_pmgbrd_shape_mismatch_raises():
    with pytest.raises(ValueError, match="broadcast"):
        pmgbrd(np.array([1.0, 2.0, 3.0]), np.array([100, 200]), 0.008)


def test_pmgbrd_crude_rate_per_100k_in_extra():
    r = pmgbrd(50.0, 1_000_000, 0.01)
    assert r.extra["crude_attributable_rate_per_100k"] > 0
    # At PAF ~ 0.3, rate ~ 1000 * 0.3 = 300 per 100k per year.
    assert 100 < r.extra["crude_attributable_rate_per_100k"] < 600


def test_pmgbrd_alias_matches():
    assert pmgbrd is pm_gemm_burden


def test_pmgbrd_custom_gemm_params_flow_through():
    # Sensitivity: steeper θ => more attributable deaths.
    r_base = pmgbrd(30.0, 100_000, 0.008)
    r_high = pmgbrd(30.0, 100_000, 0.008, theta=0.25)  # > NCD's 0.1430
    assert r_high.value > r_base.value
