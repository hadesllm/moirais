"""Tests for moirais.fn.brktm -- Berkson's bias test."""

import pytest
from moirais.fn.brktm import berkson_bias_test


class TestBerksonBias:
    def test_no_bias(self):
        res = berkson_bias_test(
            or_hospital=2.0, or_population=2.0,
            se_hospital=0.3, se_population=0.3
        )
        assert res.estimate == pytest.approx(1.0)

    def test_bias_detected(self):
        res = berkson_bias_test(
            or_hospital=4.0, or_population=1.5,
            se_hospital=0.2, se_population=0.2
        )
        assert res.extra["p_value"] < 0.05

    def test_invalid(self):
        with pytest.raises(ValueError):
            berkson_bias_test(or_hospital=-1, or_population=2.0, se_hospital=0.3, se_population=0.3)
