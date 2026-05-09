---
title: 'MOIRAIS: A Multi-Domain Scientific Computing Toolkit, with the DLRM Framework for Canadian Carceral, Police, and Oversight Data'
tags:
  - Python
  - R
  - causal inference
  - scientific experimentation
  - criminology
  - carceral analysis
  - signal processing
  - cryptography
  - spatial statistics
  - psychometrics
  - DLRM
  - Mandela Rules
  - terminal user interface
authors:
  - name: Vansh Singh Ruhela
    orcid: 0009-0004-1750-3592
    affiliation: 1
affiliations:
  - name: Independent
    index: 1
date: 9 May 2026
bibliography: paper.bib
---

# Summary

MOIRAIS (**M**ethods for **O**bservational **I**nference and
**R**obust **A**nalysis of **I**nterventions in **S**cientific
**E**xperimentation) is an open-source, dual-language (Python and R)
scientific-computing toolkit that supports observational
inference and intervention analysis across a wide range of
scientific-experimentation contexts, with sociolegal data
analysis (carceral, police, and oversight) as its named
flagship domain. The package is
home to the DLRM (Doob–Levinsky–Ruhela–Medina) framework, a
unified mathematical foundation that integrates five distinct
Canadian carceral, police, and oversight data streams under one
set of estimators [@ruhela2026dlrm]: provincial Ontario
restrictive-confinement microdata (OTIS), federal Structured
Intervention Unit reports and academic replications by
Sprott–Doob–Iftene [@sprottdoob2021torture;
@sprottdoobiftene2021iedm], the Ontario Special Investigations
Unit (police-oversight) corpus, the Toronto Police Service
open-data crime categories with the Statistics Canada Crime
Severity Index [@wallace2009csi], and federal Corrections and
Conditional Release Statistical Overview tables introduced via
Doob's *T-539-20* affidavit [@doob2020affidavit]. DLRM is
implemented in MOIRAIS as a set of RF modules — a 10-estimator
per-individual causal ensemble paired with a GEE
clustering grid, a Doob $\chi^{2}$ family for aggregate
contingency tables, a Goffmanian institutional-churn analysis
suite [@goffman1961asylums], and a Mandela Rules classifier
[@un2015mandela] that operates at both the federal level (full
out-of-cell-hour operationalization) and the provincial level
(duration-only proxy). Beyond the sociolegal flagship domain, MOIRAIS provides
general-purpose causal-inference estimators (IPW, AIPW,
double machine learning [@chernozhukov2018double],
propensity-score matching, sensitivity analysis), survey-weighted
inference [@lumley2010complex], signal processing and spectral
analysis (with applications to forensic audio and biomedical
signals), homomorphic-deconvolution methods, cryptographic
primitives, spatial statistics (Hawkes self-exciting processes,
Moran's *I*, Ripley's *K*, Getis–Ord $G^{\ast}$), stochastic
physics of crime (reaction-diffusion, Lévy flight, urban
scaling), and classical-test-theory and item-response-theory
psychometrics. The toolkit ships with 41 built-in Canadian
datasets, runs entirely from the terminal via a 10-screen
Textual TUI, and supports a multi-provider LLM chain (local
Ollama, free OllamaFreeAPI, Gemini, OpenAI-compatible) with a
vendored TurboQuant [@zandieh2026turboquant] KV-cache compression
implementation for offline inference.

# Statement of need

Sociolegal data analysis — DLRM's named home domain — sits
between criminology, statistics, and law, and is produced by disjoint government agencies under
disjoint legal frameworks. In the Canadian context alone, an
analyst comparing provincial restrictive-confinement (OTIS),
federal Structured Intervention Units (CSC), Ontario police
oversight (Special Investigations Unit), and Toronto Police
Service open data must currently stitch together five separate
data dictionaries, five release cadences, and five formats
(microdata CSV, PDF report, HTML director's report, GeoJSON,
JSON). No primary key joins these sources, and most existing
software targets one of them in isolation. MOIRAIS addresses this
gap by providing a single namespace and a single set of
estimators that span all five sources, with DLRM/RF as the
unifying mathematical layer.

The package's reach extends well beyond the sociolegal
flagship into adjacent scientific-experimentation contexts:

- **General observational inference.** Researchers in any field
  needing IPW, AIPW, double-machine learning, propensity-score
  matching, instrumental variables, regression discontinuity, or
  E-value sensitivity analysis [@vanderweele2017evalue] can use
  MOIRAIS without engaging the carceral / oversight modules.
- **Forensics and biomedical signals.** The
  `signal_processing` and `homomorphic_deconvolution` modules
  expose spectral analysis, cepstral methods, wavelets, and the
  blind-deconvolution stack relevant to forensic audio and
  biosignal applications.
- **Cryptography.** A `crypto` module collects classical and
  modern primitives suitable for teaching and for low-stakes
  research workflows.
- **Spatial statistics, statistical physics, and
  psychometrics.** Each is a self-contained module that can be
  used independently.

MOIRAIS is terminal-first by design. Public-sector analysts and
researchers in secure or air-gapped environments often lack a
graphical desktop or stable internet, and most existing causal-
inference and observational-analysis tools assume a browser-based
notebook front end. MOIRAIS's Textual TUI provides interactive
data exploration, pipeline execution, polyglot REPL (Python / R /
Shell with bidirectional variable bridging), and LLM-assisted
analysis without a web browser, and the vendored TurboQuant
KV-cache compression makes local inference practical on
consumer hardware.

# Key features

## DLRM framework (the RF modules)

The carceral, police, and oversight analyses are organised
around the DLRM framework. The RF modules, MOIRAIS's
implementation of DLRM, cover:

- An **aggregate IRR family** for Poisson- and
  negative-binomial-distributed counts with a log offset and
  fixed-effects strata.
- A **per-individual 10-estimator causal ensemble** (IPW–Hájek,
  AIPW, IRM-DML with cross-fitting, PSM, PSM with
  subclassification, AIPW with SuperLearner
  [@vanderlaan2007superlearner], PLR-DML, ATE/ATT/ATC) with
  *canonical* and *dual* (naive-arm) sensitivity formulations
  on three OTIS individual-level files.
- A **GEE clustering grid** with priority-ordered selection
  among Poisson/NB families × Exch/Indep working correlations,
  for high-cardinality groupings such as the 25 Ontario
  correctional institutions or the 5 CSC federal regions.
- A **Doob $\chi^{2}$ family** that applies Pearson $\chi^{2}$
  and Cramér's *V* to every meaningful 2-way slice of OTIS
  aggregate tables.
- A **Goffmanian institutional-churn suite** operationalising
  Goffman's [@goffman1961asylums] *total institutions* thesis
  via Gini concentration on repeat placements, Pareto vs
  log-normal AIC for embedding, joint $\chi^{2}$ for
  mortification, and intra-fiscal-year first-order Markov
  transition matrices.
- A **Mandela Rules classifier** that runs at the federal level
  with the full Sprott–Doob operationalization and at the
  provincial level with a transparent duration-only proxy. The
  cross-jurisdiction comparison reveals that Ontario provincial
  Mandela "torture"-classified proportions rise monotonically
  from 12.5 % in fiscal 2023 to 20.6 % in fiscal 2025, exceeding
  the federal SIU rate of 9.9 % that Sprott and Doob found for
  fiscal 2019–2020 [@sprottdoob2021torture].

## Crime-data spatial-temporal stack

A general Hawkes self-exciting process with exponential, gamma,
Weibull, and Lomax (power-law) excitation kernels, sinusoidal or
constant baselines, and a time-rescaling Q–Q diagnostic
[@daley2003pointprocess]. Companion spatial statistics include
Moran's *I* (global, local, bivariate), Ripley's *K*, and
Getis–Ord $G^{\ast}$. A stochastic-physics-of-crime stack adds
Short–Brantingham–D'Orsogna reaction-diffusion
[@shortbrantingham2008], Lévy-flight scaling, urban scaling
[@bettencourt2007], and Lotka–Volterra police–crime dynamics.

## Ontario SIU automation

A polite scraper, parser, normaliser, and writer for the Ontario
Special Investigations Unit director's-report corpus. Output is
a 65-column canonical CSV keyed on the SIU case number
(`YY-XXX-NNN`); seven RichResult-emitting analysers cover
demographics, decision timing, mental-health-and-race
indicators, by-police-service breakdowns, and the case-count
panel.

## General-purpose modules

Causal inference (`causal`, `effects`, `investigation`),
survey-weighted estimation (`survey`), psychometrics (`psymet` —
Cronbach's α, McDonald's ω, KMO, Bartlett, parallel analysis,
composite reliability, AVE), signal processing
(`signal_processing`, `homomorphic_deconvolution`), spatial
statistics (`spatial`), genomics utilities (`genomics`),
cryptographic primitives (`crypto`), and a 5,800+ file `fn/`
namespace of individual statistical / scientific functions.

## Terminal user interface and LLM integration

A Textual TUI with 10 screens, a multi-language REPL with
bidirectional Python / R / Shell variable bridging, and a
provider chain (local Ollama → vendored OllamaFreeAPI client →
Gemini → OpenAI-compatible → keyword-fallback) for
LLM-assisted analysis. The vendored TurboQuant
[@zandieh2026turboquant] KV-cache compression preserves the
unbiased inner-product property required for downstream causal
inference, and a pure-Python emissions tracker covers 213
countries.

# Mathematics

The DLRM framework, including the per-individual 10-estimator
ensemble, the GEE priority ordering, the Doob $\chi^{2}$ family,
the Goffmanian institutional-churn estimators, and the Mandela
Rules classifier, is developed in full in [@ruhela2026dlrm].
That paper also documents an empirical replication of all five
published $\chi^{2}$ statistics from the three Sprott–Doob–Iftene
academic reports, reproducing each value to within 0.01 of the
published number from the transcribed cell counts.

# Acknowledgements

The author thanks Prof. Beatrice Jauregui (University of
Toronto) for the network of mentorship that made this line of
work possible, and Prof. Ayobami Laniyonu (University of
Toronto) for valuable lessons over the broader period of this
work that informed the author's thinking on policing and
corrections research. The DLRM methodology lineage acknowledges
the federal context provided by Anthony N. Doob's affidavit
[@doob2020affidavit] and the four Sprott–Doob–Iftene
independent academic reports [@sprottdoob2020operation;
@sprottdoob2020covid; @sprottdoob2021torture;
@sprottdoobiftene2021iedm]. All implementation, all framework
design, and all empirical findings are the work of the framework
author.

# References
