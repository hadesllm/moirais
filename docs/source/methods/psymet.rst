Psychometric Methods
====================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

MOIRAIS provides **250+ psychometric functions** for Likert-scale questionnaire
validation, covering Classical Test Theory (CTT), Item Response Theory (IRT),
Differential Item Functioning (DIF), Confirmatory Factor Analysis (CFA),
measurement invariance, reliability variants, validity evidence, network
psychometrics, Bayesian psychometrics, scale construction, and scoring.

These methods are implemented as individual function files in ``moirais.fn/``
(short ≤7-char names) and the parent module ``moirais.psymet``.
All functions are dataset-agnostic — they work on any Likert-scale data,
not just the MAPQII questionnaire.

Key function families:

- **IRT** (20): ``irt1p``, ``irt2p``, ``irt3p``, ``irtgr``, ``irtpc``, ``irtif``, ``irttf``, ``irtic``, ``irtab``, ``irtfl``, ``irtdl``, ``irtds``, ``irtwd``, ``irtdp``, ``irtoc``, ``irtml``, ``irteap``, ``irtli``, ``irtcl``, ``irtrm``
- **DIF** (5): ``difmh``, ``diflr``, ``difef``, ``difgn``, ``difag``
- **Reliability** (15): ``kr20``, ``kr21``, ``gl1``-``gl6``, ``rglb``, ``rsem``, ``rseh``, ``rmdc``, ``rmci``, ``rcsem``, ``rirr``
- **CFA/FA** (10): ``cfa4``, ``cfabi``, ``cfafi``, ``cfaln``, ``cfars``, ``cfami``, ``cfacm``, ``efa2``, ``efart``, ``efasc``
- **Subscale** (10): ``s_ee``, ``s_ea``, ``s_ua``, ``s_er``, ``scor``, ``sdscr``, ``sconv``, ``snorm``, ``sitdl``, ``sttot``
- **Invariance** (10): ``mi_cf``, ``mi_mt``, ``mi_sc``, ``mi_st``, ``midif``, ``migen``, ``miage``, ``milat``, ``miest``, ``misum``
- **Validity** (10): ``vcnvg``, ``vdscr``, ``vhtmt``, ``vcrit``, ``vpred``, ``vknwn``, ``vmtmm``, ``vface``, ``vinc``, ``vtest``
- **Network** (10): ``netcr``, ``netst``, ``netbt``, ``netcl``, ``netei``, ``netbr``, ``netdn``, ``netcm``, ``netsb``, ``netcp``
- **Bayesian** (10): ``balph``, ``bomg``, ``birt``, ``bcfa``, ``bppc``, ``bdif``, ``bmi``, ``bfsc``, ``brcc``, ``bcomp``
- **Scale/Scoring** (20): ``itdif``-``ittab``, ``scsum``-``scrng``
- **Classical** (11): ``crba``, ``mcdo``, ``itcor``, ``adel``, ``crel``, ``ave``, ``kmo``, ``bart``, ``paran``, ``splhf``, ``idisc``

Reliability
-----------

**Cronbach's Alpha** — internal consistency coefficient:

.. math::

   \alpha = \frac{k}{k-1}\left(1 - \frac{\sum_{i=1}^k \sigma^2_{X_i}}{\sigma^2_T}\right)

where :math:`k` is the number of items, :math:`\sigma^2_{X_i}` is
the variance of item :math:`i`, and :math:`\sigma^2_T` is the total
score variance.

**McDonald's Omega Total** — model-based reliability:

.. math::

   \omega_t = 1 - \frac{\sum u_i^2}{\mathbf{1}^T R \mathbf{1}}

where :math:`u_i^2 = 1 - h_i^2` are uniquenesses from a factor model
and :math:`R` is the correlation matrix.

**Omega Hierarchical** — proportion of variance from the general factor:

.. math::

   \omega_h = \frac{(\sum \lambda_{g,i})^2}{\mathbf{1}^T R \mathbf{1}}

where :math:`\lambda_{g,i}` are first-factor loadings.

.. code-block:: python

   from moirais.psymet import crba, mcdo
   import pandas as pd

   data = pd.read_excel("data/datasets/vsr/TKARONTOMAPQ.xlsx", sheet_name="MAPQII")
   result = crba(data)
   print(f"Alpha: {result.raw:.4f} [{result.ci_lo:.4f}, {result.ci_hi:.4f}]")

   omega = mcdo(data, nf=4)
   print(f"Omega total: {omega.total:.4f}")

Factor Analysis Prerequisites
-----------------------------

**KMO** (Kaiser-Meyer-Olkin) — sampling adequacy:

.. math::

   \text{KMO} = \frac{\sum \sum_{i \neq j} r_{ij}^2}
   {\sum \sum_{i \neq j} r_{ij}^2 + \sum \sum_{i \neq j} q_{ij}^2}

where :math:`r_{ij}` are correlations and :math:`q_{ij}` are partial
correlations. Values > 0.6 are adequate; > 0.8 is "meritorious".

**Bartlett's Test of Sphericity** — tests whether the correlation
matrix differs from the identity matrix:

.. math::

   \chi^2 = -\left(n - 1 - \frac{2k+5}{6}\right) \ln|\mathbf{R}|

**Parallel Analysis** (Horn, 1965) — compares observed eigenvalues to
the 95th percentile of eigenvalues from random data with the same
dimensions. Retain factors where observed exceeds random.

Construct Validity
------------------

**Composite Reliability** (CR):

.. math::

   \text{CR} = \frac{(\sum \lambda_i)^2}{(\sum \lambda_i)^2 + \sum (1 - \lambda_i^2)}

**Average Variance Extracted** (AVE):

.. math::

   \text{AVE} = \frac{\sum \lambda_i^2}{k}

AVE ≥ 0.5 indicates adequate convergent validity.

MAPQ Validated Results
----------------------

The Modified Attitudes on Psychedelics Questionnaire (MAPQ) is a
20-item Likert (1-5) scale with 4 subscales (5 items each):

- **EE**: Epistemological Engagement (EE1-EE5)
- **EA**: Ethical Awareness (EA1-EA5)
- **UA**: Utilitarian Assessment (UA1-UA5)
- **ER**: Existential Reflection (ER1-ER5)

- **Cronbach's α** — full scale 0.943; EE 0.832; EA 0.749; UA 0.772; ER 0.822.
- **Omega total** — full scale 0.962; EE 0.836; EA 0.758; UA 0.778; ER 0.832.
- **CR (CFA)** — EE 0.838; EA 0.755; UA 0.773; ER 0.830 (full scale not modelled).
- **AVE** — EE 0.509; EA 0.398; UA 0.416; ER 0.510 (full scale not modelled).

KMO overall MSA: 0.787. Bartlett's :math:`\chi^2`: 612.22, df=190, p < 2.2e-16.

Dataset: ``data/datasets/vsr/TKARONTOMAPQ.xlsx`` (short key: ``mapq``).
