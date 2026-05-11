"""Tests for morie.fn.swdb -- Star Wars dataset loader."""

import pytest
from morie.fn.swdb import load_sw_dataset, swdb


class TestSwdb:
    def test_people_columns(self):
        df = load_sw_dataset("people")
        assert "name" in df.columns
        assert "height" in df.columns
        assert "mass" in df.columns
        assert "homeworld" in df.columns

    def test_all_tables(self):
        for table in ("people", "planets", "films", "species", "starships"):
            df = load_sw_dataset(table)
            assert len(df) > 0

    def test_unknown_table(self):
        with pytest.raises(ValueError, match="Unknown table"):
            load_sw_dataset("droids")
