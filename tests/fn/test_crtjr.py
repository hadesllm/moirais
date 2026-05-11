"""Tests for morie.fn.crtjr — R v Jordan compliance."""

import pytest
import numpy as np
from morie.fn.crtjr import court_jordan
from morie.fn._containers import CrimeResult


class TestCourtJordan:
    def test_all_compliant(self):
        r = court_jordan([100, 200, 300])
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.0)

    def test_some_over(self):
        r = court_jordan([100, 600, 700])
        assert r.n == 2
        assert r.rate == pytest.approx(2 / 3)
