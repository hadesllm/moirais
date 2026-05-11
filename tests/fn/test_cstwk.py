"""Tests for morie.fn.cstwk — custody work program."""

import pytest
import pandas as pd
from morie.fn.cstwk import custody_work_program
from morie.fn._containers import CrimeResult


class TestCustodyWorkProgram:

    def test_returns_crime(self):
        df = pd.DataFrame({"person_id": [1, 2, 3, 4], "work_program": [1, 0, 1, 0]})
        result = custody_work_program(df)
        assert isinstance(result, CrimeResult)
        assert result.rate == pytest.approx(0.5)

    def test_all_participating(self):
        df = pd.DataFrame({"person_id": [1, 2], "work_program": [1, 1]})
        result = custody_work_program(df)
        assert result.rate == pytest.approx(1.0)
