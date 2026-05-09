"""Tests for EPA 2024 PFAS Hazard Index."""

import pytest

from moirais.fn.pfashi import pfashi, pfas_hazard_index


def test_pfashi_clean_sample_is_compliant():
    r = pfashi({
        "pfoa": 1.0, "pfos": 1.0,
        "pfhxs": 2.0, "hfpo-da": 2.0, "pfna": 2.0, "pfbs": 100.0,
    })
    # HQs: 2/10 + 2/10 + 2/10 + 100/2000 = 0.2+0.2+0.2+0.05 = 0.65
    assert r.value == pytest.approx(0.65)
    assert r.extra["compliant"] is True
    assert r.extra["violation"] == []


def test_pfashi_pfoa_mcl_violation():
    r = pfashi({"pfoa": 5.0, "pfos": 1.0})
    assert r.extra["pfoa_compliance"] is False
    assert r.extra["compliant"] is False
    assert any("PFOA" in v for v in r.extra["violation"])


def test_pfashi_hi_violation():
    # PFHxS 15 ppt alone: HQ = 15/10 = 1.5 > 1
    r = pfashi({"pfhxs": 15.0})
    assert r.value == pytest.approx(1.5)
    assert r.extra["compliant"] is False
    assert any("HI" in v for v in r.extra["violation"])


def test_pfashi_hi_boundary_exactly_1():
    # PFHxS = 10 ppt gives HQ = 1.0 exactly → compliant (≤ 1)
    r = pfashi({"pfhxs": 10.0})
    assert r.value == pytest.approx(1.0)
    assert r.extra["compliant"] is True   # HI ≤ 1 is compliant


def test_pfashi_genx_is_hfpoda_synonym():
    r1 = pfashi({"hfpo-da": 5.0})
    r2 = pfashi({"genx": 5.0})
    assert r1.extra["per_compound_hq"]["hfpo-da"] \
        == pytest.approx(r2.extra["per_compound_hq"]["hfpo-da"])


def test_pfashi_case_insensitive_keys():
    r1 = pfashi({"PFOA": 2.0, "PFOS": 2.0, "PFHxS": 5.0})
    r2 = pfashi({"pfoa": 2.0, "pfos": 2.0, "pfhxs": 5.0})
    assert r1.value == pytest.approx(r2.value)


def test_pfashi_unknown_compound_raises():
    with pytest.raises(KeyError, match="Unknown PFAS"):
        pfashi({"pfos-related-thingy": 5.0})


def test_pfashi_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        pfashi({"pfoa": -1.0})


def test_pfashi_alias_matches():
    assert pfashi is pfas_hazard_index


def test_pfashi_epa_thresholds_exposed():
    r = pfashi({"pfoa": 1.0})
    t = r.extra["epa_thresholds"]
    assert t["mcl_pfoa_ppt"] == 4.0
    assert t["mcl_pfos_ppt"] == 4.0
    assert t["hi_threshold"] == 1.0
    assert t["hbwc_ppt"]["pfhxs"] == 10.0
