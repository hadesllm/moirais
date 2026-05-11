"""Test db_to_power (db2pw)."""
from morie.fn.db2pw import db_to_power, db2pw
from morie.fn._containers import DescriptiveResult


class TestDb2pw:
    def test_basic(self):
        result = db_to_power(20.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "db_to_power"
        assert abs(result.value - 100.0) < 1e-10

    def test_zero_db(self):
        result = db_to_power(0.0)
        assert abs(result.value - 1.0) < 1e-10

    def test_alias(self):
        assert db2pw is db_to_power
