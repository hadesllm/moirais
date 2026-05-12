# MA thesis — draft pasted by Ruhela 2026-05-07

> Stochastic Physics of Crime in Toronto: A Multi-Method Empirical Validation Study with the MORIE Open-Source Toolkit
> Vansh Singh Ruhela ¹,² · Yoda ¹
> ¹ HADES-LLM   ² Affiliation TBD   —   May 7, 2026

## Abstract (verbatim)

We test five canonical stochastic-physics-of-crime models against 833 217 real Toronto Police
Service (TPS) incident records spanning nine offense categories and eleven calendar years
(2014–2025) using the open-source morie analysis toolkit. For each category we fit (i) a
temporal exponential-kernel Hawkes process, (ii) a Lévy-flight Hill-MLE tail exponent on
chronological inter-incident step lengths, (iii) the Bettencourt urban-scaling exponent β across
158 Toronto neighbourhoods, (iv) a Lotka–Volterra crime–enforcement predator–prey cycle,
and (v) the Helbing–Szolnoki–Perc cooperator–predator–inspector replicator phase diagram.
We further triangulate spatial structure with DBSCAN density clustering, global and local
Moran's I, Getis–Ord G*ᵢ, the Short–D'Orsogna–Brantingham 2008 reaction–diffusion PDE,
and a Kulldorff space–time scan. All analyses ship inside a content-addressable version-control
system (clew) with a three-language port parity check, ensuring bit-identical reproducibility
on Rust, Zig and C++. Our key empirical findings: (a) Toronto's violent-crime urban scaling
is super-linear with β = 1.09 ± 0.11; (b) the Lévy-mobility tail exponent α_assault = 1.33 ± 0.00
falls in the heavy-tail regime predicted by Brockmann–Hufnagel–Geisel for human travel;
(c) Hawkes branching ratio κ is sub-critical for homicide (κ_H = 0.14) but near-critical for
property crime (κ_theft = 0.95); (d) the SDB hot-spot PDE predicts cluster counts within 22%
of empirical DBSCAN clusters at ε = 0.3 km; and (e) per-ward Moran's I is consistently
positive and significant across all categories (p < 0.001).

[full body of pasted draft preserved verbatim — see Ruhela's 2026-05-07 message]

## Issues flagged on first read (to fix during quantification pass)

1. **Hawkes Table 2 AIC column is broken** — every row reads as a hugely negative AIC
   (−537 004 for Assault, −31 238 for Homicide, etc).  These are the same daily-timestamp
   degenerate fits the new `tps_hawkes_advanced` jitter fix corrects.  Re-run with
   `tps_stochastic.hawkes_temporal_fit` after applying U(0,1)-day jitter, or use the
   new `compare_hawkes_kernels` row for the Markovian (exponential, constant) fit.

2. **Affiliation = HADES-LLM only** — drop the "TBD".  The memory rule
   (`feedback_dual_co_authorship.md`) is to use Yoda + Vansh, no model surface name.

3. **Hawkes section is single-kernel** — the new Hawkes Paper (`/papers/hawkes-paper/`)
   adds non-exponential kernels and time-varying baselines, with a measured
   ΔAIC = 141 in favour of Weibull/sinusoidal over the Markovian classical on Assault.
   The MA thesis should fold in a "Markovian vs non-Markovian" subsection rather than
   keep the M1 row at exponential-only.

4. **References missing the Kwan-Chen-Dunsmuir line** — add Kwan 2022/2023/2024/2025
   for the non-Markovian Hawkes machinery (already cited in the Hawkes Paper's bib).

5. **Some tables reference figures by `??`** — undefined cross-references in pasted
   draft; resolve once the full LaTeX is set up.

6. **Pre-2014 filter footnote** — already noted in §2.1; verify the Hawkes / Langevin /
   SARIMA fits in the existing 206 paper applied this filter (LAST_WORDS_e said yes).

7. **DMT_Imaging is NOT cloned yet** — only memory-referenced.  Not relevant to the
   crime-physics thesis directly, but the broader MORIE-coverage claim ("end-to-end
   confirmation of the entire D'Orsogna-Perc 2015 review") should not implicitly absorb
   the entheogen surface.

## Quantification plan (next pass)

Per-category numbers to verify against the codebase:

- Hawkes (μ, κ, ω, AIC, KS p) for all 9 categories — re-run with jitter
- Lévy α with bootstrap SE — current draft has α = 1.32-1.42 with SE ≈ 0; SE should be re-estimated
- Bettencourt β with R² — current draft values look right for violent (≈1.09) but Bicycle Theft β = 0.81 ± 0.24 with R² = 0.07 is a near-zero fit; flag explicitly
- Lotka-Volterra T — many categories show T = 6 283.2 yr (= 2π·10³ ≈ identifiability boundary); flag as unidentified rather than reporting
- Moran's I, DBSCAN clusters — verify against tps_spatial / tps_spatial_advanced outputs
- SDB spike count vs DBSCAN cluster count — verify the "within 22%" claim with current data
- Goffman: Hill α_churn = 1.62, Gini = 0.575, KM median 210d — verify against otis_* outputs
- LISA quadrant percentages — verify against tps_spatial_advanced
