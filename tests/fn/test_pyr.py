"""Tests for moirais.fn.pyr -- person-years at risk."""

import numpy as np
import pytest
from moirais.fn.pyr import person_years_at_risk


class TestPersonYears:
    def test_known(self):
        """3 people: 365, 730, 182 days => ~1 + 2 + 0.498 PY."""
        entry = np.array(["2020-01-01", "2020-01-01", "2020-01-01"], dtype="datetime64[D]")
        exit_ = np.array(["2021-01-01", "2022-01-01", "2020-07-01"], dtype="datetime64[D]")
        res = person_years_at_risk(entry, exit_)
        assert res.measure == "Person-years"
        assert res.estimate == pytest.approx(365 / 365.25 + 731 / 365.25 + 182 / 365.25, rel=0.01)

    def test_n(self):
        """n should equal number of subjects."""
        entry = np.array(["2020-01-01", "2020-06-01"], dtype="datetime64[D]")
        exit_ = np.array(["2020-12-31", "2020-12-31"], dtype="datetime64[D]")
        res = person_years_at_risk(entry, exit_)
        assert res.n == 2

    def test_empty_raises(self):
        """Empty arrays should raise."""
        with pytest.raises(ValueError):
            person_years_at_risk(np.array([], dtype="datetime64[D]"), np.array([], dtype="datetime64[D]"))
