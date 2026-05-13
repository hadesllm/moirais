# First paper template

Drop this scaffold into your manuscript and replace the bracketed sections with your own results. Citations are pre-filled to the morie papers.

---

# [Your study title]

**[Your name]**, [your affiliation]
**Corresponding author**: [your email]

## Abstract

[200 words — restate question, briefly summarise design, headline result with effect size + CI, note 1–2 design caveats.]

**Keywords**: [3–5 keywords]

## Background

[3–5 paragraphs framing your research question and citing prior work. End with your specific hypothesis, stated formally.]

## Methods

Analyses were conducted in `morie` v0.5.0 [@Ruhela2026MoriePy; @Ruhela2026MorieR], a multi-domain scientific computing toolkit for observational inference. The Multilevel Reconciliation Methodology (MRM) framework backing morie's primary modules is documented separately [@Ruhela2026MRM].

### Data

[State your sample. If you used a morie-shipped dataset, name it: "We analysed the Canadian Postsecondary Alcohol and Drug Use Survey (CPADS) 2021-2022 microdata, accessed via morie's bundled loader." If you used your own data, describe it.]

### Design

[Describe your causal contrast, units of analysis, treatment / outcome / covariates. If you ran a power calculation, mention the target effect size and the sample size you ended up with.]

### Statistical analysis

We applied the [MODULE_NAME] module of morie, which [REPLACE_WITH_MODULE_DESCRIPTION].

[For specific modules, copy-paste one of these methods sentences and tailor it:]

- **`power-design`**: Sample-size and power calculations followed the survey-weighted two-proportion formula implemented in morie's `power-design` module, calibrated against G*Power [REF] using the cross-reference table in `power_gpower_reference_two_group.csv`.
- **`descriptive-statistics`**: Prevalence estimates were computed with the design weight (the `weight` column in CPADS) and survey-design–robust standard errors.
- **`frequentist-inference`**: Confidence intervals are Wilson intervals for binomial proportions [REF]; effect sizes are reported as Cohen's *h* for two-proportion contrasts and as odds ratios for logistic models.
- **`bayesian-inference`**: Posterior summaries assume a Beta(1, 1) (uniform) prior for proportions, with credible intervals computed from posterior quantiles.
- **`logistic-models`**: Logistic regression was fit with the survey weight; interaction terms and SMOTE oversampling for class imbalance were applied as preplanned in `logistic_smote_*.csv`.
- **`mrm_*` (causal inference)**: The MRM module composed ten estimators (IPW Hájek, AIPW RRZ doubly-robust, double machine learning IRM, propensity-score matching, …); details and theoretical motivation are in [@Ruhela2026MRM].

Reproducibility: outputs are reproducible by running `morie run-module [MODULE] --output-dir <dir>` on morie v0.5.0; the analysis CSVs are available in our supplementary materials.

## Results

[Insert your headline number with CI in the first sentence. Then walk through the supporting tables. The CSVs morie produced map roughly to:]

| morie output | Corresponds to |
|---|---|
| `power_summary.csv` | Table 1: design and sample-size recommendation |
| `binomial_summaries_survey_weighted.csv` | Table 2: prevalence with 95% CI |
| `frequentist_effect_sizes.csv` | Table 3: effect sizes |
| `frequentist_hypothesis_tests.csv` | Table 4: test statistics + p-values |
| `logistic_odds_ratios.csv` | Table 5: adjusted odds ratios |

[Reference each result with a sentence and the CSV column it draws on. Example: "Heavy drinking in the past 30 days was reported by 22.3% of respondents (95% CI [21.4, 23.2]; `binomial_summaries_survey_weighted.csv`)."]

## Discussion

[3–5 paragraphs: what the result means, how it relates to prior work, limitations, what comes next.]

## Acknowledgments

[Acknowledge funding, computing infrastructure, and collaborators. If you used morie's bundled synthetic data, state so explicitly so readers don't mistake it for real-population analyses.]

## References

```bibtex
@software{Ruhela2026MoriePy,
  author    = {Ruhela, Vansh Singh},
  title     = {morie: Multi-domain Open Research and Inferential Estimation in {Python}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20096350},
  url       = {https://doi.org/10.5281/zenodo.20096350}
}

@software{Ruhela2026MorieR,
  author    = {Ruhela, Vansh Singh},
  title     = {morie: Multi-domain Open Research and Inferential Estimation in {R}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20111233},
  url       = {https://doi.org/10.5281/zenodo.20111233}
}

@article{Ruhela2026MRM,
  author  = {Ruhela, Vansh Singh},
  title   = {{MRM}: Multilevel Reconciliation Methodology --
             A multi-source statistical foundation for {Canadian}
             carceral, police, and oversight data},
  year    = {2026},
  doi     = {10.5281/zenodo.20096075},
  url     = {https://doi.org/10.5281/zenodo.20096075}
}
```

---

## How to use this template

```bash
# Get a fresh copy to fill in:
morie generate-template --module power-design --out my-paper.md

# Then open it in any editor — Markdown means no special software needed.
```

If you've never written a methods section before: that's fine. Start with the bullets above for your module, replace `[MODULE_NAME]` with what you ran, and put your numbers into the Results table. The structure is conventional enough that a reader can follow even a first draft.
