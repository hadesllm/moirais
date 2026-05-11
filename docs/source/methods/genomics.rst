Population Genetics
===================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides population genetics functions for analyzing genetic variation,
population structure, and genotype-phenotype associations. All functions are
dataset-agnostic with column names as keyword parameters.

Sequence-Level Metrics
----------------------

- ``gc`` — GC content. Proportion of guanine + cytosine in a DNA sequence.
- ``maf`` — minor allele frequency. Frequency of the less common allele at a locus.
- ``hw`` — Hardy-Weinberg equilibrium. Chi-squared test for HWE departure (observed vs expected genotype counts).

.. code-block:: python

   from morie.fn import gc, maf, hw

   gc_ratio = gc("ATGCGCTATGCGC")
   print(f"GC content: {gc_ratio:.3f}")   # 0.615

   freq = maf(genotypes=[0, 0, 1, 1, 2, 0, 1])
   print(f"MAF: {freq:.3f}")

   hwe = hw(observed=[45, 40, 15])  # AA, Aa, aa counts
   print(f"HWE chi2={hwe.statistic:.2f}, p={hwe.p_value:.4f}")

Population Differentiation
--------------------------

- ``fst`` — fixation index. Weir-Cockerham :math:`F_{ST}` estimator for population differentiation.
- ``tajd`` — Tajima's D. Neutrality test comparing pairwise diversity to segregating sites.

.. code-block:: python

   from morie.fn import fst, tajd

   result = fst(allele_freqs_pop1, allele_freqs_pop2,
                n_pop1=100, n_pop2=120)
   print(f"Fst = {result.statistic:.4f}")

   taj = tajd(segregating_sites=42, n_sequences=50,
              pairwise_diffs=18.5)
   print(f"Tajima's D = {taj.statistic:.3f}, p = {taj.p_value:.4f}")

Linkage Disequilibrium
----------------------

- ``ld`` — linkage disequilibrium. D, D', and r-squared between two loci.
- ``ldmat`` — LD matrix. Pairwise r-squared matrix for a set of SNPs.

.. code-block:: python

   from morie.fn import ld

   result = ld(genotypes_snp1, genotypes_snp2)
   print(f"D' = {result.d_prime:.3f}, r2 = {result.r_squared:.3f}")

Genome-Wide Association Studies
-------------------------------

- ``gwas`` — GWAS scan. Per-SNP association test (linear or logistic) with multiple-testing correction.
- ``prs`` — polygenic risk score. Weighted sum of risk alleles using GWAS summary statistics.

.. code-block:: python

   from morie.fn import gwas, prs

   results = gwas(genotype_matrix, phenotype, covariates=None,
                  model="linear", correction="bonferroni")
   sig = results[results["p_adj"] < 0.05]
   print(f"Significant SNPs: {len(sig)}")

   scores = prs(genotypes, effect_sizes, risk_alleles)
   print(f"Mean PRS: {scores.mean():.3f}")

**GWAS pipeline:**

1. Quality control: MAF filter, HWE filter, missingness filter
2. Association testing: per-SNP regression (``gwas``)
3. Multiple testing correction: Bonferroni or Benjamini-Hochberg
4. Visualization: Manhattan plot, QQ plot (via ``holo_*`` functions)
5. Risk prediction: polygenic risk scores (``prs``)

Epidemiological Applications
----------------------------

Population genetics functions integrate with MORIE's epidemiological
toolkit for public health genomics:

- **Pharmacogenomics**: MAF and HWE checks for drug-metabolizing enzyme
  variants across population subgroups
- **Disease surveillance**: Fst for tracking pathogen population structure
  across geographic regions
- **Health equity**: PRS calibration across ancestry groups to avoid
  differential prediction accuracy

All functions return standardized result objects (``GenomicsResult``
dataclass from ``_containers.py``) with ``statistic``, ``p_value``,
and method-specific fields.

References
----------

.. [Weir1984] Weir, B.S. & Cockerham, C.C. (1984). Estimating F-Statistics
   for the Analysis of Population Structure. *Evolution*, 38(6), 1358-1370.

.. [Tajima1989] Tajima, F. (1989). Statistical Method for Testing the Neutral
   Mutation Hypothesis by DNA Polymorphism. *Genetics*, 123(3), 585-595.

.. [Purcell2007] Purcell, S. et al. (2007). PLINK: A Tool Set for
   Whole-Genome Association and Population-Based Linkage Analyses.
   *American Journal of Human Genetics*, 81(3), 559-575.
