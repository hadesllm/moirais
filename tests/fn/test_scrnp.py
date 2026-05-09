"""Tests for moirais.fn.scrnp -- Screening test properties."""

import pytest
from moirais.fn.scrnp import screening_properties


class TestScreening:
    def test_known(self):
        res = screening_properties(tp=90, fp=10, fn=10, tn=90)
        assert res.extra["sensitivity"] == pytest.approx(0.9)
        assert res.extra["specificity"] == pytest.approx(0.9)

    def test_ppv_npv(self):
        res = screening_properties(tp=90, fp=10, fn=10, tn=90)
        assert res.extra["PPV"] == pytest.approx(0.9)
        assert res.extra["NPV"] == pytest.approx(0.9)

    def test_invalid(self):
        with pytest.raises(ValueError):
            screening_properties(tp=10, fp=0, fn=-1, tn=10)
