"""Test power_to_db (pwrdb)."""
import numpy as np
from moirais.fn.pwrdb import power_to_db, pwrdb
from moirais.fn._containers import DescriptiveResult


class TestPwrdb:
    def test_basic(self):
        result = power_to_db(100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "power_to_db"
        assert abs(result.value - 20.0) < 1e-10

    def test_unity(self):
        result = power_to_db(1.0)
        assert abs(result.value) < 1e-10

    def test_alias(self):
        assert pwrdb is power_to_db
