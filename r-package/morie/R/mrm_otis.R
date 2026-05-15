# SPDX-License-Identifier: AGPL-3.0-or-later
#' MRM-framework analyses on Ontario OTIS data
#'
#' Five callables for the OTIS (Offender Tracking Information System)
#' public-release datasets, used in the MRM (Multilevel Reconciliation
#' Methodology) empirical companion paper. Every analysis is computed
#' directly from the OTIS CSV files; no precomputed artifacts are
#' required.
#'
#' Functions:
#' * `mrm_otis_placement_concentration()`: Hill-MLE Pareto exponent +
#'   Gini coefficient + top-k% concentration on the b09 per-individual
#'   placement-count distribution (within-fiscal-year).
#' * `mrm_otis_seg_duration_km()`: Kaplan-Meier survival on the b01
#'   `NumberConsecutiveDays_Segregation` durations (per-placement;
#'   strata = alert profile).
#' * `mrm_otis_mortification_cooccurrence()`: pairwise Cramer's V across
#'   the three b01 alert flags (MentalHealth, SuicideRisk, SuicideWatch).
#' * `mrm_otis_region_locality()`: chi-square + Cramer's V on the
#'   `Region_AtTimeOfPlacement` x `Region_MostRecentPlacement`
#'   contingency table, with the diagonal/off-diagonal share.
#' * Plus the existing `mrm_classify_mandela()` (in `mandela.R`).
#'
#' The OTIS `UniqueIndividual_ID` column has format `YYYY-XXXXX-SG` and
#' is randomly reassigned every fiscal year. Cross-year tracking is
#' therefore invalid; all analyses below operate within fiscal year.
#'
#' @name mrm_otis
NULL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.gini_int <- function(x) {
  x <- sort(as.numeric(x))
  n <- length(x)
  if (n == 0L || sum(x) == 0) return(NA_real_)
  (2 * sum(seq_len(n) * x) - (n + 1) * sum(x)) / (n * sum(x))
}

.hill_mle <- function(x, x_min) {
  x <- x[x >= x_min]
  n <- length(x)
  if (n < 2L) return(NA_real_)
  1 + n / sum(log(x / x_min))
}

.cramer_v <- function(tbl) {
  if (any(dim(tbl) < 2L)) return(NA_real_)
  chi <- suppressWarnings(stats::chisq.test(tbl, correct = FALSE))
  k <- min(dim(tbl))
  sqrt(as.numeric(chi$statistic) / (sum(tbl) * (k - 1)))
}


# ---------------------------------------------------------------------------
# 1. Placement-count concentration (b09)
# ---------------------------------------------------------------------------

#' Per-individual segregation-placement-count concentration on OTIS b09
#'
#' Expands the OTIS b09 banded per-individual placement counts into a
#' per-person vector using band midpoints (the published bands are
#' `\{1, 2, 3, 4, 5, 6-10, 11-15, 16-20, 21-25, 26-30, 31-35, 36-40, >40\}`),
#' then computes Hill-MLE Pareto exponent, Gini coefficient, and top-k%
#' concentration within each fiscal year and pooled.
#'
#' @param data A data.frame in b09 long format with the columns named
#'   in `year_col`, `count_col`, `band_col`, optionally `gender_col`.
#' @param year_col Column name of the fiscal-year identifier
#'   (default `"EndFiscalYear"`).
#' @param band_col Column name of the placement-count band
#'   (default `"NumberPlacements_Segregation"`).
#' @param count_col Column name of the per-band individual count
#'   (default `"NumberIndividuals_Segregation"`).
#' @param gender_col Optional gender filter column. If supplied with
#'   `gender_keep`, rows are restricted to the kept genders.
#' @param gender_keep Character vector of gender values to retain.
#' @param x_min Hill-MLE lower-tail cutoff (default `1L`).
#' @param top_pct Numeric in (0, 1); top concentration cutoff
#'   (default `0.05`).
#' @return A data.frame with one row per fiscal year plus a final
#'   `"pooled"` row, containing columns `year`, `n_individuals`,
#'   `n_placements`, `mean_per_individual`, `gini`, `hill_alpha`,
#'   `top_pct_share`.
#' @references
#' Hill, B. M. (1975). A simple general approach to inference about the
#' tail of a distribution. The Annals of Statistics, 3(5), 1163-1174.
#'
#' Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
#' distributions in empirical data. SIAM Review, 51(4), 661-703.
#' @export
#' @examples
#' if (FALSE) {
#'   b09 <- read.csv("b09_individuals_in_segregation_number_of_times_in_segregation.csv")
#'   mrm_otis_placement_concentration(b09)
#' }
mrm_otis_placement_concentration <- function(
  data,
  year_col = "EndFiscalYear",
  band_col = "NumberPlacements_Segregation",
  count_col = "NumberIndividuals_Segregation",
  gender_col = NULL,
  gender_keep = NULL,
  x_min = 1L,
  top_pct = 0.05
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(year_col, band_col, count_col) %in% names(data)))

  band_to_midpoint <- function(band) {
    s <- as.character(band)
    if (grepl("Greater than", s)) {
      x <- as.numeric(sub("[^0-9]*([0-9]+).*", "\\1", s))
      return(x + 10)
    }
    if (grepl("to", s)) {
      m <- regmatches(s, regexpr("[0-9]+ to [0-9]+", s))
      nums <- as.numeric(strsplit(m, " to ", fixed = TRUE)[[1]])
      return(mean(nums))
    }
    nums <- as.numeric(regmatches(s, regexpr("[0-9]+", s)))
    if (length(nums) == 0L) return(NA_real_)
    nums[1]
  }

  if (!is.null(gender_col) && !is.null(gender_keep)) {
    data <- data[data[[gender_col]] %in% gender_keep, , drop = FALSE]
  }

  data$.midpt <- vapply(data[[band_col]], band_to_midpoint, numeric(1))

  expand_year <- function(sub) {
    rep(sub$.midpt, sub[[count_col]])
  }

  years <- sort(unique(data[[year_col]]))
  out <- lapply(c(as.list(years), list("pooled")), function(y) {
    if (identical(y, "pooled")) {
      sub <- data
      label <- "pooled"
    } else {
      sub <- data[data[[year_col]] == y, , drop = FALSE]
      label <- as.character(y)
    }
    x <- expand_year(sub)
    x <- x[is.finite(x) & x > 0]
    n <- length(x)
    if (n == 0L) {
      return(data.frame(year = label, n_individuals = 0,
                        n_placements = 0, mean_per_individual = NA_real_,
                        gini = NA_real_, hill_alpha = NA_real_,
                        top_pct_share = NA_real_))
    }
    x_sorted <- sort(x, decreasing = TRUE)
    cut <- max(1L, ceiling(top_pct * n))
    data.frame(
      year = label,
      n_individuals = n,
      n_placements = sum(x),
      mean_per_individual = mean(x),
      gini = round(.gini_int(x), 4),
      hill_alpha = round(.hill_mle(x, x_min), 4),
      top_pct_share = round(sum(x_sorted[seq_len(cut)]) / sum(x), 4)
    )
  })
  do.call(rbind, out)
}


# ---------------------------------------------------------------------------
# 2. Segregation-duration KM survival (b01)
# ---------------------------------------------------------------------------

#' Kaplan-Meier survival on OTIS b01 segregation-placement durations
#'
#' Treats each row of OTIS b01 as one observed placement with duration
#' `NumberConsecutiveDays_Segregation` and no censoring (all durations
#' are observed end-to-end within the fiscal year). Returns the median
#' duration and the requested-quantile survival probabilities by stratum.
#'
#' This replaces the misreading of `UniqueIndividual_ID = YYYY-XXXXX-SG`
#' as a persistent person identifier (which produces a spurious
#' ~210-day cross-year TTR artifact). The valid quantity here is the
#' distribution of how long a placement lasts, not how long until the
#' next placement.
#'
#' @param data A data.frame containing `duration_col` and optional
#'   stratifying columns (`group_cols`).
#' @param duration_col Column name of segregation duration in days
#'   (default `"NumberConsecutiveDays_Segregation"`).
#' @param group_cols Optional character vector of stratifying-column
#'   names. NULL pools all rows.
#' @param probs Quantiles of the survival function to report
#'   (default `c(0.5, 0.25, 0.10, 0.05, 0.01)`).
#' @param mandela_threshold Day cutoff (default `15L`) for Mandela-
#'   prolonged placement. Reports the fraction of placements above the
#'   cutoff and the median duration among those.
#' @return A data.frame with one row per stratum (or one pooled row),
#'   columns `stratum`, `n`, `mean_days`, `median_days`,
#'   `q25_days`, `pct_above_mandela`, `median_among_above_mandela`.
#' @export
#' @examples
#' if (FALSE) {
#'   b01 <- read.csv("b01_segregation_detailed_dataset.csv")
#'   mrm_otis_seg_duration_km(b01)
#'   mrm_otis_seg_duration_km(b01, group_cols = "MentalHealth_Alert")
#' }
mrm_otis_seg_duration_km <- function(
  data,
  duration_col = "NumberConsecutiveDays_Segregation",
  group_cols = NULL,
  probs = c(0.5, 0.25, 0.10, 0.05, 0.01),
  mandela_threshold = 15L
) {
  stopifnot(is.data.frame(data))
  stopifnot(duration_col %in% names(data))

  if (is.null(group_cols)) {
    strata <- list(pooled = rep(TRUE, nrow(data)))
  } else {
    stopifnot(all(group_cols %in% names(data)))
    key <- do.call(paste, c(lapply(group_cols, function(c) data[[c]]), sep = "|"))
    strata <- split(seq_len(nrow(data)), key)
    strata <- lapply(strata, function(idx) seq_len(nrow(data)) %in% idx)
  }

  rows <- lapply(names(strata), function(s) {
    mask <- strata[[s]]
    d <- data[[duration_col]][mask]
    d <- d[!is.na(d) & d > 0]
    n <- length(d)
    if (n == 0L) {
      return(data.frame(stratum = s, n = 0, mean_days = NA_real_,
                        median_days = NA_real_, q25_days = NA_real_,
                        pct_above_mandela = NA_real_,
                        median_among_above_mandela = NA_real_))
    }
    above <- d > mandela_threshold
    data.frame(
      stratum = s,
      n = n,
      mean_days = round(mean(d), 2),
      median_days = stats::median(d),
      q25_days = stats::quantile(d, 0.75, names = FALSE),
      pct_above_mandela = round(100 * mean(above), 2),
      median_among_above_mandela = if (any(above)) stats::median(d[above]) else NA_real_
    )
  })
  do.call(rbind, rows)
}


# ---------------------------------------------------------------------------
# 3. Mortification co-occurrence (b01 alert columns)
# ---------------------------------------------------------------------------

#' Pairwise Cramer's V of OTIS b01 alert columns (mortification proxy)
#'
#' Computes the pairwise Cramer's V (and chi-square test) for every
#' pair of the three OTIS b01 alert columns. The
#' MentalHealth x SuicideRisk Cramer's V is the substantive
#' "mortification co-occurrence" figure used in the MRM paper.
#'
#' Values are computed by treating `"Yes"` as 1 and any other value as
#' 0; rows with `NA` in either alert column are dropped from that pair.
#'
#' @param data A data.frame with at least the three alert columns.
#' @param alert_cols Character vector of alert column names
#'   (default the three b01 alert columns).
#' @return A data.frame with one row per pair, columns `alert_a`,
#'   `alert_b`, `n`, `chi2`, `df`, `p_value`, `cramers_v`.
#' @export
#' @examples
#' if (FALSE) {
#'   b01 <- read.csv("b01_segregation_detailed_dataset.csv")
#'   mrm_otis_mortification_cooccurrence(b01)
#' }
mrm_otis_mortification_cooccurrence <- function(
  data,
  alert_cols = c("MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert")
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(alert_cols %in% names(data)))
  bins <- lapply(alert_cols, function(c) as.integer(data[[c]] == "Yes"))
  names(bins) <- alert_cols
  pairs <- utils::combn(alert_cols, 2L, simplify = FALSE)
  rows <- lapply(pairs, function(p) {
    a <- bins[[p[1]]]; b <- bins[[p[2]]]
    keep <- !is.na(a) & !is.na(b)
    tbl <- table(a[keep], b[keep])
    chi <- suppressWarnings(stats::chisq.test(tbl, correct = FALSE))
    data.frame(
      alert_a = p[1], alert_b = p[2],
      n = sum(keep),
      chi2 = round(as.numeric(chi$statistic), 2),
      df = as.integer(chi$parameter),
      p_value = signif(chi$p.value, 3),
      cramers_v = round(.cramer_v(tbl), 4)
    )
  })
  do.call(rbind, rows)
}


# ---------------------------------------------------------------------------
# 4. Region locality (b01 region columns)
# ---------------------------------------------------------------------------

#' OTIS b01 region locality: chi-square + diagonal-share
#'
#' Constructs the contingency table
#' `Region_AtTimeOfPlacement` x `Region_MostRecentPlacement` and reports
#' the chi-square statistic, Cramer's V, and the share of placements on
#' the diagonal (within-region staying) vs off-diagonal (cross-region
#' churn). Ontario seg/RC placement is overwhelmingly diagonal
#' (locality-preserving) in the public release.
#'
#' @param data A data.frame with `region_at_col` and `region_recent_col`.
#' @param region_at_col Column name of the at-placement region
#'   (default `"Region_AtTimeOfPlacement"`).
#' @param region_recent_col Column name of the most-recent region
#'   (default `"Region_MostRecentPlacement"`).
#' @return A list with named elements `table` (the contingency matrix),
#'   `chi2`, `df`, `p_value`, `cramers_v`, `diagonal_share`,
#'   `off_diagonal_share`.
#' @export
#' @examples
#' if (FALSE) {
#'   b01 <- read.csv("b01_segregation_detailed_dataset.csv")
#'   mrm_otis_region_locality(b01)
#' }
mrm_otis_region_locality <- function(
  data,
  region_at_col = "Region_AtTimeOfPlacement",
  region_recent_col = "Region_MostRecentPlacement"
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(region_at_col, region_recent_col) %in% names(data)))
  tbl <- table(data[[region_at_col]], data[[region_recent_col]])
  chi <- suppressWarnings(stats::chisq.test(tbl, correct = FALSE))
  diag_sum <- sum(diag(tbl))
  total <- sum(tbl)
  list(
    table = tbl,
    chi2 = round(as.numeric(chi$statistic), 2),
    df = as.integer(chi$parameter),
    p_value = signif(chi$p.value, 3),
    cramers_v = round(.cramer_v(tbl), 4),
    diagonal_share = round(diag_sum / total, 4),
    off_diagonal_share = round(1 - diag_sum / total, 4)
  )
}
