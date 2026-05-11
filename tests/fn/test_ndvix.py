"""Tests for NDVI green-space exposure RR."""

import math

import numpy as np
import pytest

from morie.fn.ndvix import ndvi_exposure_rr, ndvix


def test_ndvix_reference_gives_rr_one():
    r = ndvix(0.2, reference_ndvi=0.2)
    assert r.value == pytest.approx(1.0)


def test_ndvix_0_1_delta_matches_base_rr():
    r = ndvix(0.3, reference_ndvi=0.2, outcome="all_cause")
    assert r.value == pytest.approx(0.96, abs=1e-4)


def test_ndvix_is_protective_greener_is_lower_rr():
    r_low = ndvix(0.1, reference_ndvi=0.1)
    r_high = ndvix(0.5, reference_ndvi=0.1)
    # 0.96^4 ≈ 0.8493
    assert r_high.value < r_low.value
    assert r_high.value == pytest.approx(0.96 ** 4, abs=1e-4)


def test_ndvix_depression_outcome_has_stronger_effect():
    r_all = ndvix(0.4, reference_ndvi=0.2, outcome="all_cause")
    r_dep = ndvix(0.4, reference_ndvi=0.2, outcome="depression")
    # Depression RR per 0.1 = 0.85 (stronger than 0.96 all-cause)
    assert r_dep.value < r_all.value


def test_ndvix_ci_brackets_point():
    r = ndvix(0.4, reference_ndvi=0.1)
    assert r.extra["rr_95lo"] < r.value < r.extra["rr_95hi"]


def test_ndvix_array_input():
    N = np.array([0.1, 0.2, 0.3, 0.5])
    r = ndvix(N, reference_ndvi=0.1)
    rrs = r.extra["rr"]
    # Monotone decreasing (greener ⇒ protective ⇒ RR lower)
    for i in range(1, len(rrs)):
        assert rrs[i] < rrs[i - 1]


def test_ndvix_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ndvix(1.5)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ndvix(-1.5)


def test_ndvix_rejects_bad_reference():
    with pytest.raises(ValueError, match="reference_ndvi"):
        ndvix(0.3, reference_ndvi=2.0)


def test_ndvix_unknown_outcome_raises():
    with pytest.raises(KeyError, match="Unknown outcome"):
        ndvix(0.3, outcome="anxiety")


def test_ndvix_alias_matches():
    assert ndvix is ndvi_exposure_rr


def test_ndvix_outcome_normalization():
    # "self-rated-mh" / "self_rated_mh" / "Self Rated MH" all equivalent
    r1 = ndvix(0.3, outcome="self_rated_mh")
    r2 = ndvix(0.3, outcome="self-rated-mh")
    r3 = ndvix(0.3, outcome="Self Rated MH")
    assert r1.value == pytest.approx(r2.value)
    assert r2.value == pytest.approx(r3.value)
