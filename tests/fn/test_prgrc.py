"""Tests for moirais.fn.prgrc — program recidivism."""

import pytest
import numpy as np
from moirais.fn.prgrc import program_recidivism
from moirais.fn._containers import ESRes


class TestProgramRecidivism:
    def test_lower_rate(self):
        rng = np.random.default_rng(42)
        p = rng.binomial(1, 0.2, 200)
        c = rng.binomial(1, 0.4, 200)
        r = program_recidivism(p, c)
        assert isinstance(r, ESRes)
        assert r.estimate < 0

    def test_too_small(self):
        with pytest.raises(ValueError):
            program_recidivism([1], [0])
