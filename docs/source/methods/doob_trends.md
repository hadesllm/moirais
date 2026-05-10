# Doob T-539-20 Federal Court affidavit replication

*Part of {doc}`index` — MOIRAIS's statistical-methods reference.*

`moirais.doob_trends` replicates the analytical contribution of
Prof. Anthony N. Doob's expert-witness affidavit in *Canadian Civil
Liberties Association et al. v. The Attorney General of Canada*,
Federal Court file **T-539-20** (Application Record Vol. 3 of 5,
pp. 778-795).

The affidavit was filed in 2020 in support of the application for
COVID-19-driven prisoner release; it presents 4 figures + 3 tables
arguing that imprisonment rates are **decoupled** from crime rates
and that conditional release is empirically safe.

## Three CCRSO 2018 tables hardcoded

The module hardcodes the three numerical tables from Doob's
affidavit, sourced from Public Safety Canada's *Corrections and
Conditional Release Statistical Overview 2018*:

- `CCRSO_TABLE1_RELEASES` — affidavit Table 1: 5-year average
  annual conditional releases by type.
- `CCRSO_TABLE2_FLOW` — affidavit Table 2: annual prisoner counts,
  admissions, deaths, releases (2013/14-2017/18).
- `CCRSO_TABLE3_AGE` — affidavit Table 3: age distribution
  (Canada adult population vs CSC custody).

```python
from moirais.doob_trends import (
    analyze_doob_table1_releases,
    analyze_doob_table2_flow,
    analyze_doob_table3_age_overrepresentation,
    analyze_doob_full_affidavit,
)

r = analyze_doob_table1_releases()
# Headline: violent revocation rate < 1% across all release types

r = analyze_doob_table3_age_overrepresentation()
# Headline: age 60+ is 29.5% of Canadian adult pop but only 9.3%
# of CSC custody (IRR_custody = 0.32)
```

## Decoupling test

Doob's central thesis (§§ 16-21 of the affidavit) is that
imprisonment rates and crime rates are not consistently correlated.
`decoupling_test()` tests this with Pearson r + Pettitt change-point:

```python
from moirais.doob_trends import decoupling_test

decoupling_test(crime_rate_series, imprisonment_rate_series,
                 years=range(1960, 2018))
# RichResult with r_pearson, two-sided p, Pettitt change-point per
# series.
```

`pettitt_changepoint(series)` exposes the Pettitt 1979 non-
parametric change-point test as a standalone helper.

## Note on Figures 1-4

Doob's affidavit also contains four time-series figures (Canadian
crime rate 1960-2018, Canadian imprisonment rate 1960-2018, Canada
vs US imprisonment 1950-2017, Canada vs US homicide 1961-2017).
These plots use StatsCan CANSIM data + Pastore-Maguire US Sourcebook
of Criminal Justice Statistics. The data series themselves are not
bundled with moirais; users supply their own time-series for use
with `decoupling_test()`.

## Position in the MRM stack

The Doob national-aggregate analyses complement the OTIS provincial
evidence:

- **Federal national** (this module): Tables 1-3 + decoupling.
- **Federal SIU regional** (`moirais.sprott_doob`): Mandela
  classifier + regional Pacific-vs-Ontario disparities.
- **Provincial individual + aggregate** (`moirais.otis_*`): Ruhela
  formulations ensemble on OTIS Ontario data.

`analyze_ruhela_master()` §5 surfaces key rows from this module.

## Citation

```python
from moirais.siuiap import cite
cite("doob_t_539_20_2020")
# Anthony N. Doob (2020). Affidavit of Anthony N. Doob — Federal
# Court of Canada, T-539-20. Federal Court of Canada.
```
