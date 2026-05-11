"""Tests for morie.fn.migrt -- net migration rate."""

import pytest
from morie.fn.migrt import net_migration_rate


class TestNetMigration:
    def test_positive(self):
        res = net_migration_rate(immigrants=500, emigrants=200, population=100000)
        assert res.estimate == pytest.approx(3.0)

    def test_negative(self):
        res = net_migration_rate(100, 500, 100000)
        assert res.estimate < 0
