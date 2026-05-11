"""Tests for morie.fn.cstin — custody incident rate."""

import pytest
from morie.fn.cstin import custody_incident_rate


class TestCustodyIncidentRate:
    def test_returns_dict(self, otis_df):
        result = custody_incident_rate(otis_df)
        assert isinstance(result, dict)

    def test_keys(self, otis_df):
        result = custody_incident_rate(otis_df)
        assert "events" in result
        assert "person_days" in result
        assert "rate_per_1000" in result

    def test_rate_positive(self, otis_df):
        result = custody_incident_rate(otis_df)
        assert result["rate_per_1000"] >= 0

    def test_custom_cols(self, otis_df):
        df = otis_df.rename(columns={"D": "event", "sentence_days": "time"})
        result = custody_incident_rate(df, event_col="event", time_col="time")
        assert result["events"] >= 0
