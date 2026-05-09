"""Tests for moirais.fn.sirdm -- SIR with age-structured demographics."""

import pytest
from moirais.fn.sirdm import sir_age_demographics


class TestSIRDM:
    def test_runs(self):
        res = sir_age_demographics()
        assert res.model == "SIR-AGE"
        assert len(res.t) == 1000

    def test_r0_positive(self):
        res = sir_age_demographics()
        assert res.R0 > 0

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            sir_age_demographics(gamma=-1)
