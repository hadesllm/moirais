"""Tests for moirais.fn.mtovh — vehicle type."""

import pytest
import pandas as pd
from moirais.fn.mtovh import mto_vehicle_type
from moirais.fn._containers import DescriptiveResult


class TestVehicleType:
    def test_basic(self):
        df = pd.DataFrame({"vehicle_type": ["Car"] * 10 + ["Truck"] * 5, "n_crashes": [1] * 15})
        r = mto_vehicle_type(df)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["by_type"]["Car"] == 10

    def test_no_crash_col(self):
        df = pd.DataFrame({"vehicle_type": ["Car", "Car", "Truck"]})
        r = mto_vehicle_type(df)
        assert r.extra["by_type"]["Car"] == 2
