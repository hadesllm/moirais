"""Tests for difag — DIF by age group."""
import numpy as np
from moirais.fn.difag import difag

def test_difag_basic(mapq_df):
    # Need binary age groups for MH DIF
    mapq_df = mapq_df.copy()
    mapq_df["age_binary"] = np.where(mapq_df["age_group"].isin(["18-24","25-34"]), "young", "old")
    result = difag(mapq_df, age_col="age_binary")
    assert result is not None


def test_cheatsheet():
    from moirais.fn.difag import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
