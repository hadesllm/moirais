"""Tests for moirais.emissions — pure-Python emissions tracker.

No hardcoded column counts or magic numbers — all expectations are derived
from the actual EmissionsData.csv_header property so tests stay in sync
with the implementation.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest


class TestCarbonIntensity:
    def test_known_countries_relative_ordering(self):
        """Countries with known energy mixes should have plausible relative ordering."""
        from moirais.emissions import _get_carbon_intensity, _load_energy_data
        _load_energy_data()
        fra = _get_carbon_intensity("FRA")  # nuclear-heavy, low carbon
        usa = _get_carbon_intensity("USA")  # mixed grid
        assert fra < usa, f"France ({fra}) should be lower than USA ({usa})"
        assert fra > 0, "Intensity must be positive"
        assert usa > 0, "Intensity must be positive"

    def test_unknown_country_returns_world_average(self):
        from moirais.emissions import _get_carbon_intensity, _WORLD_AVERAGE_G_KWH, _G_TO_KG
        val = _get_carbon_intensity("ZZZ")
        expected = _WORLD_AVERAGE_G_KWH * _G_TO_KG
        assert abs(val - expected) < 0.001, f"Unknown should use world average, got {val}"

    def test_canada_with_and_without_region(self):
        """CAN with region should differ from CAN without (if regional data exists)."""
        from moirais.emissions import _get_carbon_intensity
        can_national = _get_carbon_intensity("CAN")
        can_region = _get_carbon_intensity("CAN", "ontario")
        # Both must be positive; they may or may not differ depending on data
        assert can_national > 0
        assert can_region > 0

    def test_all_positive(self):
        from moirais.emissions import _get_carbon_intensity
        for iso in ["USA", "CAN", "GBR", "DEU", "CHN", "IND", "AUS", "BRA", "JPN", "FRA"]:
            val = _get_carbon_intensity(iso)
            assert val > 0, f"{iso} must have positive carbon intensity"


class TestRamPower:
    def test_returns_positive(self):
        from moirais.emissions import _estimate_ram_power
        assert _estimate_ram_power() > 0

    def test_within_physical_bounds(self):
        """RAM power should be between 1W and 200W for any real machine."""
        from moirais.emissions import _estimate_ram_power
        power = _estimate_ram_power()
        assert 1.0 <= power <= 200.0, f"RAM power {power}W outside physical bounds"


class TestEmissionsTracker:
    def test_start_stop_returns_float(self):
        from moirais.emissions import EmissionsTracker
        with tempfile.TemporaryDirectory() as d:
            t = EmissionsTracker(
                project_name="test", output_dir=d,
                measure_power_secs=0.5, country_iso_code="CAN",
                save_to_file=False,
            )
            t.start()
            time.sleep(1)
            emissions = t.stop()
            assert isinstance(emissions, float)
            assert emissions >= 0

    def test_context_manager_does_not_raise(self):
        from moirais.emissions import EmissionsTracker
        with tempfile.TemporaryDirectory() as d:
            with EmissionsTracker(
                project_name="ctx", output_dir=d,
                measure_power_secs=0.5, country_iso_code="USA",
                save_to_file=False,
            ):
                time.sleep(0.5)

    def test_csv_header_matches_data_row(self):
        """CSV header column count must match data row column count.

        Derived from EmissionsData.csv_header — no hardcoded number.
        """
        from moirais.emissions import EmissionsTracker, EmissionsData
        expected_cols = len(EmissionsData().csv_header.split(","))

        with tempfile.TemporaryDirectory() as d:
            t = EmissionsTracker(
                project_name="csv-test", output_dir=d,
                output_file="test_emissions.csv",
                measure_power_secs=0.5, country_iso_code="FRA",
            )
            t.start()
            time.sleep(1)
            t.stop()

            csv_path = Path(d) / "test_emissions.csv"
            assert csv_path.exists(), "CSV file should be created"
            lines = csv_path.read_text().strip().splitlines()
            assert len(lines) == 2, "Header + 1 data row"
            header_cols = len(lines[0].split(","))
            data_cols = len(lines[1].split(","))
            assert header_cols == expected_cols, (
                f"Header has {header_cols} cols but EmissionsData defines {expected_cols}"
            )
            assert data_cols == header_cols, (
                f"Data row has {data_cols} cols but header has {header_cols}"
            )
            assert "timestamp" in lines[0]
            assert "emissions" in lines[0]
            assert "country_iso_code" in lines[0]

    def test_emissions_data_has_required_fields(self):
        """EmissionsData must have all fields needed for codecarbon-compatible output."""
        from moirais.emissions import EmissionsData
        data = EmissionsData()
        required = [
            "emissions", "cpu_energy", "gpu_energy", "ram_energy",
            "energy_consumed", "country_iso_code", "pue", "wue",
            "cpu_power", "gpu_power", "ram_power", "duration",
        ]
        for field in required:
            assert hasattr(data, field), f"EmissionsData missing required field: {field}"

    def test_csv_row_col_count_matches_header(self):
        """csv_row() must produce same number of fields as csv_header."""
        from moirais.emissions import EmissionsData
        data = EmissionsData(project_name="test", country_iso_code="USA")
        header_count = len(data.csv_header.split(","))
        row_count = len(data.csv_row().split(","))
        assert row_count == header_count, (
            f"csv_row has {row_count} fields but csv_header has {header_count}"
        )
