"""Shared fixtures for moirais.fn tests.

'He who would learn to fly one day must first learn to stand and walk. — Friedrich Nietzsche'
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def linear_df(rng):
    """Linear data: y = 2*x + 0.5 + noise. Known slope=2, intercept=0.5."""
    n = 200
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.5 + rng.standard_normal(n) * 0.3
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture()
def binary_df(rng):
    """Binary outcome data for logistic/causal tests."""
    n = 300
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    treatment = rng.binomial(1, prob, n)
    outcome = 0.5 * treatment + 0.3 * x1 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome})


@pytest.fixture()
def grouped_df(rng):
    """Data with 3 groups for ANOVA / grouped tests."""
    n_per = 50
    a = rng.normal(10, 2, n_per)
    b = rng.normal(12, 2, n_per)
    c = rng.normal(14, 2, n_per)
    return pd.DataFrame({
        "value": np.concatenate([a, b, c]),
        "group": ["A"] * n_per + ["B"] * n_per + ["C"] * n_per,
    })


@pytest.fixture()
def missing_df(rng):
    """Data with ~10% missing values."""
    n = 200
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "z": rng.choice(["a", "b", "c"], n),
    })
    mask = rng.random(n) < 0.1
    df.loc[mask, "x"] = np.nan
    mask2 = rng.random(n) < 0.1
    df.loc[mask2, "y"] = np.nan
    return df


@pytest.fixture()
def contingency_df(rng):
    """Contingency table data for chi-squared / Fisher tests."""
    n = 200
    return pd.DataFrame({
        "group": rng.choice(["A", "B"], n, p=[0.6, 0.4]),
        "outcome": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })


@pytest.fixture()
def survival_df(rng):
    """Survival data with time and event indicator."""
    n = 100
    time = rng.exponential(5, n)
    censor_time = rng.exponential(8, n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(int)
    x = rng.standard_normal(n)
    return pd.DataFrame({"time": observed_time, "event": event, "x": x})


@pytest.fixture()
def timeseries_df(rng):
    """AR(1) time series: y_t = 0.7 * y_{t-1} + noise."""
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + rng.standard_normal()
    return pd.DataFrame({"y": y, "t": np.arange(n)})


@pytest.fixture()
def otis_df(rng):
    """Synthetic OTIS correctional placement data (500 records, 100 individuals)."""
    n_ids = 100
    records_per = 5
    n = n_ids * records_per
    regions = ["Central", "Eastern", "Northern", "Toronto", "Western"]
    ages = ["18 to 24", "25 to 49", "50+"]
    genders = ["Male", "Female"]
    years = [2018, 2019, 2020, 2021, 2022]

    ids = np.repeat(np.arange(1, n_ids + 1), records_per)
    return pd.DataFrame({
        "unique_individual_id": ids,
        "end_fiscal_year": rng.choice(years, n),
        "region": rng.choice(regions, n),
        "age_group": rng.choice(ages, n, p=[0.3, 0.5, 0.2]),
        "gender": rng.choice(genders, n, p=[0.8, 0.2]),
        "alert_mental_health": rng.binomial(1, 0.25, n),
        "alert_suicide_risk": rng.binomial(1, 0.15, n),
        "alert_suicide_watch": rng.binomial(1, 0.05, n),
        "facility_type": rng.choice(["Provincial Correctional Centre",
                                     "Community Supervision"], n),
        "sentence_days": rng.integers(30, 730, n),
        "start_date": pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D"),
        "Y": rng.standard_normal(n),
        "D": rng.binomial(1, 0.4, n),
    })


@pytest.fixture()
def mapq_df(rng):
    """Synthetic MAPQII psychometric data (200 respondents, 20 Likert items)."""
    n = 200
    items = {}
    # 4 subscales of 5 items each, with moderate inter-item correlation
    for sub, prefix in [("EE", "EE"), ("EA", "EA"), ("UA", "UA"), ("ER", "ER")]:
        latent = rng.standard_normal(n)
        for i in range(1, 6):
            raw = latent + rng.standard_normal(n) * 0.8
            # Convert to 1-5 Likert
            items[f"{prefix}{i}"] = np.clip(np.round(raw * 0.8 + 3), 1, 5).astype(int)
    df = pd.DataFrame(items)
    df["gender"] = rng.choice(["Male", "Female"], n)
    df["age_group"] = rng.choice(["18-24", "25-34", "35-49", "50+"], n)
    return df


@pytest.fixture()
def mapq_binary_df(rng):
    """Synthetic binary (dichotomous) item data for KR-20/KR-21."""
    n = 200
    k = 10
    # Items with varying difficulty
    difficulties = np.linspace(0.3, 0.8, k)
    data = {}
    for i in range(k):
        data[f"item_{i+1}"] = rng.binomial(1, difficulties[i], n)
    return pd.DataFrame(data)


@pytest.fixture()
def signal_1khz(rng):
    """1 kHz sine + 50 Hz mains + noise at 8000 Hz."""
    fs = 8000
    t = np.arange(0, 1.0, 1 / fs)
    x = (np.sin(2 * np.pi * 1000 * t)
         + 0.5 * np.sin(2 * np.pi * 50 * t)
         + rng.standard_normal(len(t)) * 0.1)
    return x, fs


@pytest.fixture()
def ecg_synthetic(rng):
    """Synthetic ECG with known R-peaks at 1 kHz, ~75 bpm."""
    fs = 1000
    n = int(5.0 * fs)
    ecg = np.zeros(n)
    r_peaks = np.arange(500, n, 800)
    for pk in r_peaks:
        ecg += 1.5 * np.exp(-0.5 * ((np.arange(n) - pk) / 10) ** 2)
    ecg += rng.standard_normal(n) * 0.05
    return ecg, fs, r_peaks
