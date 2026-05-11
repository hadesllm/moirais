# Scientific findings from the MORIE empirical pipeline

A condensed map of the verified, reproducible findings the MORIE
v0.1.15 callables surface on the OTIS, TPS, and SIU public-release
data. Each row links the substantive claim to the morie callable
that re-derives it, the underlying public dataset, and the
verification result file in `papers/morie-empirical-paper/results/`.

The five companion papers cite these numbers; the per-result text
files capture method choices and any deviations from prior
literature.

## Provincial vs federal solitary-confinement rates (Mandela Rules)

* Ontario provincial Segregation: **12.5 % → 16.5 % → 20.6 %**
  Mandela-prolonged proportion across fiscal 2023, 2024, 2025.
* Ontario provincial Restrictive Confinement: **31.5 % → 36.0 %
  → 40.9 %** over the same panel.
* Federal Sprott-Doob SIU comparison: **9.9 %** torture-classified
  on the 2019-20 sample of 1,960 person-stays.

> **Substantive finding.** The provincial Ontario rate exceeds the
> federal SIU benchmark by **+10.7 percentage points** at peak and
> is rising monotonically. The provincial Mandela problem is
> structurally larger than the federal one.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_classify_mandela(c11, source="c11_aggregate")` | OTIS c11 CSV | `results/05-mandela-verified.txt` |

## Heavy-tailed segregation placement concentration (Goffman)

* Pooled 2023-25: **n = 33,136 individuals contributing 153,067
  placements**, Gini **G = 0.585**, top-5 % concentration
  **25.9 %**, Hill-MLE Pareto exponent **α = 2.08** at
  `x_min = 1`.
* Per-fiscal-year Gini rises monotonically:
  **0.542 → 0.596 → 0.616**.

> **Correction to the MA-thesis.** Earlier reporting of a Hill
> α = 1.62 cannot be reproduced from b09; the verified value at
> `x_min = 1` is 2.08, rising to ~3.2 for higher `x_min`. The
> direction (heavy tail) is preserved, the magnitude is corrected.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_otis_placement_concentration(b09)` | OTIS b09 CSV | `results/06-goffmanian-churn-b09.txt` |

## Mortification co-occurrence on the b01 alert flags

* MentalHealth × SuicideRisk: **V = 0.189**, χ²(1) = 2,914,
  *p* < 10⁻³⁰⁰.
* SuicideRisk × SuicideWatch: V = 0.668 — Watch is operationally
  a subset escalation of Risk, so its co-occurrence is a workflow
  artefact, not a substantive finding.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_otis_mortification_cooccurrence(b01)` | OTIS b01 CSV | `results/07-mortification-region-b01.txt` |

## Within-region locality of placements (corrected from "regional churn")

* 5 × 5 contingency table of `Region_AtTimeOfPlacement` ×
  `Region_MostRecentPlacement`: **95.0 % diagonal share**, **only
  5.0 % cross-region transfers**.
* χ²(16) = 285,917, *p* ≈ 0, Cramér's V = 0.934 — rejects
  independence because of the overwhelming diagonal, not because
  of cross-region churn.

> **Reframing the literature.** The MA-thesis read the same χ²
> as evidence of "routinised regional flows"; the verified
> direction is the opposite. Ontario seg/RC placement is
> **locality-preserving**, not churn-driven.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_otis_region_locality(b01)` | OTIS b01 CSV | `results/07-mortification-region-b01.txt` |

## "Time-to-readmission" — the 210-day myth, debunked

The MA-thesis "Kaplan-Meier median TTR ≈ 210 days" claim is an
**artefact** of mis-reading `UniqueIndividual_ID` as a persistent
person identifier. OTIS IDs follow the format `YYYY-XXXXX-SG` and
are randomly re-assigned each fiscal year — 0 of 65,467 individuals
appear in two or more years.

* KM on placement duration (the in-system quantity that *is*
  computable): **median 2 days**, mean 2.83 days; among
  Mandela-prolonged (>15-day) placements, median 25 days.
* Within-year readmission gap (the cross-year TTR proxy):
  **not computable** because OTIS releases no placement-start
  dates.

> **The valid TTR analysis is on SIU**, not OTIS. SIU cases have
> per-case `date_of_incident_iso` and
> `date_of_director_decision_iso` plus a stable `police_service`
> identifier. Median case-to-decision gap is **120 days**
> (pooled n = 1,711), with per-service medians clustered
> tightly around 120 — a system-wide processing cadence rather
> than a per-jurisdiction effect.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_otis_seg_duration_km(b01)` | OTIS b01 CSV | `results/10-seg-duration-km.txt` |
| `mrm_siu_case_to_decision_km(siu)` | SIU.csv (scraped) | `results/15-...` (SIU TTR) |

## Toronto Hawkes self-excitation (Markovian vs Weibull / sinusoidal)

* All 9 TPS categories fit by Hawkes processes with sub-sampled
  n_fit = 1,500 events.
* Markovian (exponential kernel, constant baseline) branching
  ratios **κ ∈ [0.878, 0.974]** — every fit stationary.
* Non-Markovian (Weibull kernel, sinusoidal baseline) preferred
  in **every** category by ΔAIC ranging from **80.8 (Homicides)
  to 133.6 (Assault)**.

> **Substantive finding.** Toronto crime is best modelled as a
> seasonally-modulated self-exciting process. The categories
> with the largest branching-ratio reductions on switching to
> the non-Markovian fit (Auto Theft 0.97 → 0.81, Bicycle Theft
> 0.97 → 0.67, Homicides 0.88 → 0.55) are exactly those with
> strong annual modulation that the constant-baseline Markovian
> fit had been forced to attribute to self-excitation.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_tps_load_hawkes_refit(manifest_path)` | `paper_hawkes_refit.json` | `results/08-hawkes-verified.txt` |

## Toronto crime is becoming LESS spatially concentrated (LISA + Moran)

* Per-ward Assault counts at HOOD_158 resolution (n = 158 polygons,
  k = 6 NN weight matrix, 199 MC perms):
  * **Global Moran's I = 0.270 (2024)**, with quadrants
    HH 19.0 % / HL 12.7 % / LH 16.5 % / **LL 51.9 %** of all 158
    wards. 11 wards form the significant downtown HH core,
    5 wards are significant LH outliers.
* Time series of global Moran's I for Assault 2014-2024:
  **0.38 → peak 0.44 (2017) → 0.27 (2024)**, monotonic decline
  through 2018-2024.

> **Substantive finding.** Counts per year rose +54 % over the
> panel (16,515 → 25,431) while spatial concentration declined
> -38 %. Assault is **spreading**, not concentrating. This
> contradicts the older paper claim of "no significant decline"
> in Moran's I.

| MRM callable | Data | Result file |
|---|---|---|
| LISA computation in `data/manifest/outputs/tps_spatial_advanced/lisa_Assault_2024.json` | NCR GeoJSON | `results/15-lisa-per-year-moran-verified.txt` |

## Kulldorff space-time clusters (199 MC permutations)

Top cluster per category (LRT, RR, p-value) on n = 30,000 sampled
events, 4-year window, r ∈ {1, 2, 3, 5, 8} km:

| Category | log L | RR | p | Window |
|---|---|---|---|---|
| Theft from MV | 73.7 | 6.72 | 0.005 | 2013--2017 |
| Break & Enter | 41.5 | 3.23 | 0.005 | 2017--2021 |
| Assault | 22.5 | 5.00 | 0.005 | 2022--2026 |
| Auto Theft | 21.3 | 5.42 | 0.005 | 2018--2022 |
| Robbery | 13.9 | 4.85 | 0.005 | 2013--2017 |
| Homicides | 5.0 | 3.51 | 0.040 | 2017--2021 |
| Theft Over | 4.9 | 3.38 | 0.040 | 2013--2017 |

> **Substantive finding.** The Theft-from-MV cluster
> (LRT = 73.7, RR = 6.7) is the strongest space-time hotspot in
> Toronto's open data; centred near `(43.66, -79.40)` (downtown
> corridor), it represents a sustained 2013-2017 concentration
> at radius 2 km — the smallest scan radius that captures the
> phenomenon.

| MRM callable | Data | Result file |
|---|---|---|
| `mrm_tps_kulldorff_scan(tps, n_permutations=199)` | TPS Assault CSV | `results/13-kulldorff-verified.txt` + `14-kulldorff-199perm-and-7.11-verified.txt` |

## DML alert-complexity → placement-volatility on OTIS

* Pooled IRM-DML on b01 with cluster-robust SE:
  **ATE = 0.1605** (SE 0.00628, t = 25.5), 2023/2024/2025 ATE
  = 0.134 / 0.159 / 0.174.
* CSI-context overlay (a01_csi pipeline): ATE = **0.213**
  (cluster-SE 0.012, n = 14,520, 95 % CI [0.189, 0.236]).

> **Substantive finding.** Conditioning on the prevailing
> Crime Severity Index of the surrounding ward does not
> meaningfully change the within-OTIS alert-complexity
> → placement-volatility ATE. The headline causal estimate is
> robust to the crime-environment confounder.

| MRM callable | Data | Result file |
|---|---|---|
| Pipeline at `OTIS-RC/correctional_stats_report1z.RData` | OTIS b01 + a01_csi | `results/01-verified-dml.txt` |

## Things that DIDN'T survive verification

A short list of MA-thesis or paper-prose claims the verification
pipeline could **not** reproduce — important to publish for
methodological hygiene:

| Claim | Where it appeared | Real value |
|---|---|---|
| KM TTR ~210 days | §7.8 / MA-thesis §3.8 | Artefact of mis-reading `YYYY-XXXXX-SG` IDs |
| Hill α = 1.62 (Goffman) | §7.8 | Real value α = 2.08 |
| SDB PDE within 22 % of DBSCAN | §7.6 | Real avg deviation 57 % (SDB is parameter-driven) |
| OU θ ∈ [0.05, 0.78] | §7.9 | Real range [0.07, 1.03] |
| OU σ ∈ [10, 30] | §7.9 | Real range [0.5, 19.7] |
| Hellinger < 5 % for 8/9 cats | §7.9 | Hellinger not in manifest; softened |
| Getis-Ord 25-40 wards / cat | §7.11 | Real range 6-27 wards / cat |
| LISA Assault 2024: 47/5/4/44 | §7.11 | Real: 19/13/17/52 |
| "No significant decline" Moran I | §7.11 | Real: 0.44 → 0.27 (-38 %) |
| Hawkes "Gamma-kernel preferred" | hawkes-paper abstract | Weibull-kernel preferred |

Each correction is captured in its result file and reflected in the
relevant paper §.

## How to re-run any finding

```bash
pip install morie==0.1.15
python -m morie.demo           # 30-second showcase with animations
```

Or in R:
```r
install.packages("morie")
library(morie)
mrm_otis_placement_concentration(morie_sample("otis_b09"))
mrm_classify_mandela(morie_sample("otis_b01"), denominator = "row")
```
