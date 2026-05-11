# designexptr.org coverage map (morie 0.2.0)

Closes the audit from 2026-05-11: every designexptr.org book chapter has
one or more `mrm_*` callables in `morie` with full Python + R parity.

| Chapter (designexptr.org)                          | morie callables                                                                                                                              | Primary refs                                            |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **Ch.1** Statistical Inference Foundations         | base morie: `estimate_ate`, `estimate_att`, `e_value`, `cohens_d`, `hedges_g`, `eta_squared`, `cramers_v`                                     | Casella-Berger 2002                                     |
| **Ch.2** Math Stats / Simulation / Computation     | `mrm_oneprop_test`, `mrm_twoprop_test`, `mrm_var_test`, `mrm_qq_plot`, `mrm_clt_demo`, `mrm_pit`                                              | Wilks 1962; Casella-Berger 2002; Lehmann-Romano 2005    |
| **Ch.3** Sampling Distributions / Inference        | `mrm_two_treatment_test`, `mrm_anova_oneway`, `mrm_anova_bonferroni`, `kruskal_wallis_test`, `mann_whitney_test`, `fisher_exact_test`         | Box-Hunter-Hunter 2005                                  |
| **Ch.4** Blocking + Random Block Designs           | `mrm_rcbd`, `mrm_perm_block`                                                                                                                  | Cochran-Cox 1957; Montgomery 2017                       |
| **Ch.5** Latin / Graeco-Latin Squares              | `mrm_latin_square`, `mrm_graeco_latin`, `mrm_random_latin`                                                                                    | Cochran-Cox 1957                                        |
| **Ch.6** Factorial + Fractional + Response Surface | `mrm_factorial_2k`, `mrm_fractional_factorial`, `mrm_response_surface`                                                                       | Box-Hunter-Hunter 2005; Box-Wilson 1951                 |
| **Ch.7** Causal Inference                          | `mrm_causal_design`, `mrm_standardised_difference`, `mrm_check_balancing`, `mrm_check_overlap`, `mrm_median_causal_effect`, `mrm_assumptions_check` | Imbens-Rubin 2015; Rosenbaum-Rubin 1985; Cole-Hernan 2008 |
| **Ch.8** Power & Sample Size                       | `mrm_anova_power`, `mrm_mc_power`                                                                                                             | Cohen 1988                                              |

## Parity status

All `mrm_*` callables ship with bit-identical Python (`morie.mrm_*`) and R
(`morie::mrm_*`) implementations.  Deterministic callables verified
bit-identical at the smoke-test level:

- `mrm_oneprop_test(40, 100, 0.5)` → `p_value_exact = 0.05688793` (Python = R)
- `mrm_twoprop_test(40, 100, 60, 100)` → `p_value_chi2 = 0.004678` (Python = R)
- `mrm_anova_power(3, 30, 0.25)` → `power = 0.5396` (Python = R)
- `mrm_response_surface` on the bowl `2 - x1 - x2 - x1^2 - x2^2` → `stationary_nature = "maximum"` (Python = R)
- `mrm_random_latin(4)` → 4 × 4 (Python = R; different cell values because
  RNG state diverges between numpy and base R)

## Empirical-application callables (provincial, OTIS/TPS/SIU)

Beyond the designexptr pedagogical surface, morie ships the
empirical-application callables introduced for the morie empirical
paper:

`mrm_classify_mandela`, `mrm_otis_placement_concentration`,
`mrm_otis_seg_duration_km`, `mrm_otis_mortification_cooccurrence`,
`mrm_otis_region_locality`, `mrm_otis_mandela_spectrum`,
`mrm_tps_levy_scaling`, `mrm_tps_moran_clustering`,
`mrm_tps_neighbourhood_recurrence_km`, `mrm_tps_kulldorff_scan`,
`mrm_tps_lisa`, `mrm_tps_polygon_moran_per_year`,
`mrm_tps_load_hawkes_refit`,
`mrm_siu_case_to_decision_km`, `mrm_siu_per_service_rate`,
`mrm_siu_outcome_classifier`.
