Acknowledgments
===============

AI assistance
-------------

MORIE was developed with substantial assistance from frontier AI
assistants. The author retains full responsibility for the code, the
methods, and the scientific claims. AI assistance accelerated
implementation but does not change the attribution of the work.

This disclosure is included for transparency; it does not transfer
authorship, copyright, or licensing obligations away from the human
author. Where AI-generated code reproduces material from training
data verbatim, the upstream licence governs that material; the
author has reviewed the source for any such cases.

Anthropic Claude
~~~~~~~~~~~~~~~~

Anthropic's Claude family — Opus, Sonnet, and Haiku across the 4.x
generation — was used extensively throughout development for code
generation, refactoring, documentation, code review, and design
discussions. Use was supported by the Anthropic research-credit
program.

Project URL: https://www.anthropic.com/claude

Google Gemini and Vertex AI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Google's Gemini 2.5 models — Pro and Flash — running on the
Google Cloud Vertex AI platform were used extensively for additional
code generation, cross-checking Claude-generated code, multi-modal
data analysis, and prototype evaluation. Use was supported by the
Google research-credit program.

Project URLs:

- https://deepmind.google/technologies/gemini/
- https://cloud.google.com/vertex-ai

The Anthropic and Google research-credit programs are compute-
allocation programs; they do not constitute endorsement of MORIE
by either company.

Funding and infrastructure
--------------------------

- Anthropic — Claude API research credits.
- Google — Gemini / Vertex AI research credits.

Mentorship and expert review
----------------------------

The author thanks **Glenn McNamara** — a 35-year career with the
Ontario Government — for his methodological mentorship. Glenn
introduced the author to foundational distribution theory and the
applied-statistics intuition for administrative data that grounds
much of this framework. He is the **M** (catalyst, for McNamara) in
**MRM (Multilevel Reconciliation Methodology)** — the framework that
powers the OTIS / SIU / TPS analyses across this package.

The author thanks **Prof. Angela Zorro Medina**, Centre for
Criminology and Sociolegal Studies, University of Toronto, who is
the author's **supervisor**, **methodological instructor**, the
**domain-expert reviewer** of the preliminary methodological
approach, and a **knowledge user** of the framework. Prof. Medina
is the **M** (supervisor & reviewer) in MRM. The methodological
lineage MRM follows is established in her work on anti-gang
legislation:

   Zorro Medina, Á. (2023). *The Effect of Anti-Gang Laws on Crime
   and Social Control.* University of Chicago Job Market Paper.
   https://azorromedina.com/wp-content/uploads/2023/08/JMP_ZorroMedina_28_08_23H.pdf

Specifically, MRM inherits from her JMP:

- a **staggered two-way-fixed-effects** identification strategy
  with formal **leads-and-lags Granger-causality** diagnostics for
  the parallel-trends assumption (Athey & Imbens, 2022;
  Callaway & Sant'Anna, 2021; Goodman-Bacon, 2021);
- an explicitly **multi-source data-integration** pattern — her
  five U.S. sources (FBI SRS/NIBRS, BJS BJSPS, FBI UCR, BLS, state
  annotated criminal codes) are structurally analogous to MRM's
  five Canadian sources (OTIS / SIU IAP / Ontario SIU / TPS / CCRSO);
- the **deterrence / routine-activities / certainty** mechanism
  categorisation that grounds individual estimands in sociolegal
  theory rather than statistical fit alone;
- the **inequality-effects-of-criminal-law-expansions** framing
  (§2.3 of the JMP) that connects empirical estimands to racial
  and social inequality in carceral outcomes.

Her substantive review of the OTIS Major Research Paper insisted on
quantitatively-grounded sociolegal mechanism, on a negative-binomial
mixed model with regional-cluster random intercepts as the principal
aggregate-level estimator, and on a disciplined empirical structure.
That review directly shapes the methodology reported here.

Data acknowledgments
--------------------

Several MRM analyses use Statistics Canada and Health Canada Public
Use Microdata Files (PUMFs):

- **CCS** — Canadian Cannabis Survey (Health Canada, annual since
  2018; multiple cycles 2018-2024).
- **CSADS** — Canadian Student Alcohol and Drugs Survey
  (2021-22, 2023-24; previously *Canadian Student Tobacco, Alcohol
  and Drugs Survey* / CSTADS, 2014-2022; previously *Youth Smoking
  Survey* / YSS, before 2014).
- **CSUS** — Canadian Substance Use Survey (2023, 2025).
- **CADS** — Canadian Alcohol and Drugs Survey
  (2019; https://doi.org/10.25318/132500052021001-eng).
- **CPADS** — Canadian Postsecondary Education Alcohol and Drug
  Use Survey (2019-20, 2021-22).

Although the analyses use Statistics Canada data, the analyses,
interpretations, and conclusions are those of the author and do
not represent the views of Statistics Canada. Public Health Agency
of Canada (PHAC) and Canadian Institute for Health Information
(CIHI) aggregates are used under the same standard disclaimer.
Ontario open data (OTIS, A01-RCDD release; via
``data.ontario.ca``) and Toronto Police Service open data are
acknowledged with the same standard disclaimer.
