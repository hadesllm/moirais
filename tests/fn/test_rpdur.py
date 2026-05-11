"""Tests for rpdur — placement duration."""
import pandas as pd
from morie.fn.rpdur import rplace_duration

def test_rpdur_basic(otis_df):
    otis_df["end_date"] = otis_df["start_date"] + pd.to_timedelta(otis_df["sentence_days"], unit="D")
    result = rplace_duration(otis_df, start_col="start_date", end_col="end_date")
    assert isinstance(result, (pd.DataFrame, dict))


def test_cheatsheet():
    from morie.fn.rpdur import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
