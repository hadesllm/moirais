# SPDX-License-Identifier: AGPL-3.0-or-later
#' Generalised predictive-policing disparity audit
#'
#' R parity for the Python `morie.fairness.predpol` module. A
#' clean-room, city-agnostic reimplementation of the district-level
#' analysis of the SciencesPo *Predictive-policing-Chicago* project
#' (Lacherade, Szabo, Krikava & Aeby, 2021): rank areas by the risk an
#' algorithm predicts, rank them by their realised outcome rate, and
#' test whether the disagreement tracks the areas' demographic
#' composition.
#'
#' Functions:
#' * `predpol_aggregate_areas()`: roll per-record data up to one row
#'   per area.
#' * `predpol_calibration_audit()`: Spearman calibration plus a
#'   per-group mean rank gap (the over-/under-prediction signal).
#' * `predpol_score_disparity()`: descriptive per-group risk-score
#'   summary with a one-way ANOVA.
#'
#' Written from the project's published methodology; no code copied
#' (that repository carries no licence and is not redistributable).
#'
#' @name frns_predpol
NULL


.frns_worst_abs_named <- function(x) {
  # name of the element with the largest absolute value.
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_character_)
  names(x)[which.max(abs(x))]
}


#' Aggregate per-record predictive-policing data to one row per area
#'
#' @param area Area identifier for each record.
#' @param risk Predicted risk score for each record.
#' @param outcome Realised-outcome indicator/count for each record.
#' @param group Optional protected attribute per record; the per-area
#'   majority value becomes that area's group label.
#' @param population Optional area population: a named numeric vector
#'   (`area -> population`) or a per-record vector. When given, the
#'   outcome rate is per 10,000 inhabitants; otherwise it is the mean
#'   outcome per record.
#' @return A named list: `areas`, `mean_risk`, `outcome_rate`, `group`,
#'   `n_records`.
#' @export
#' @examples
#' agg <- predpol_aggregate_areas(
#'   area = c("a", "a", "b", "b"), risk = c(10, 20, 30, 40),
#'   outcome = c(1, 0, 1, 1))
#' agg$mean_risk      # 15 35
#' agg$outcome_rate   # 0.5 1.0
predpol_aggregate_areas <- function(area, risk, outcome, group = NULL,
                                    population = NULL) {
  if (length(area) != length(risk) || length(area) != length(outcome)) {
    stop("area, risk and outcome must be the same length", call. = FALSE)
  }
  area <- as.character(area)
  risk <- as.numeric(risk)
  outcome <- as.numeric(outcome)
  areas <- sort(unique(area))

  mean_risk <- vapply(areas, function(a) mean(risk[area == a]), numeric(1))
  counts <- vapply(areas, function(a) sum(outcome[area == a]), numeric(1))
  n_records <- vapply(areas, function(a) sum(area == a), integer(1))

  if (is.null(population)) {
    outcome_rate <- vapply(areas, function(a) mean(outcome[area == a]),
                           numeric(1))
  } else {
    if (!is.null(names(population))) {
      pops <- as.numeric(population[areas])
    } else {
      if (length(population) != length(area)) {
        stop("population vector must be the same length as area",
             call. = FALSE)
      }
      population <- as.numeric(population)
      pops <- vapply(areas, function(a) population[area == a][1], numeric(1))
    }
    outcome_rate <- ifelse(pops > 0, counts / pops * 10000, NA_real_)
  }

  maj <- NULL
  if (!is.null(group)) {
    if (length(group) != length(area)) {
      stop("group must be the same length as area", call. = FALSE)
    }
    group <- as.character(group)
    maj <- vapply(areas, function(a) {
      tab <- table(group[area == a])
      names(tab)[which.max(tab)]
    }, character(1))
  }

  list(areas = areas, mean_risk = unname(mean_risk),
       outcome_rate = unname(outcome_rate),
       group = if (is.null(maj)) NULL else unname(maj),
       n_records = unname(n_records))
}


#' Audit whether an algorithm's area risk ranking matches realised outcomes
#'
#' Ranks areas by predicted risk and by realised outcome rate (rank 1 =
#' highest), forms `rank_gap = outcome_rank - risk_rank` per area
#' (positive = over-predicted), and averages the gap within each group.
#' A Spearman correlation summarises overall calibration.
#'
#' @param areas Area identifiers (one per area).
#' @param mean_risk Mean predicted risk per area.
#' @param outcome_rate Realised outcome rate per area.
#' @param group Majority protected-attribute label per area.
#' @return A named list: `value` (worst per-group mean gap), `spearman`,
#'   `spearman_pvalue`, `group_rank_gap`, `worst_group`, `rank_gap`,
#'   `warnings`, `interpretation`.
#' @export
#' @examples
#' res <- predpol_calibration_audit(
#'   areas = c("d1", "d2", "d3", "d4", "d5", "d6"),
#'   mean_risk = c(90, 80, 70, 30, 20, 10),
#'   outcome_rate = c(10, 20, 30, 70, 80, 90),
#'   group = c("X", "X", "X", "Y", "Y", "Y"))
#' res$group_rank_gap$X   # 3  (group X over-predicted)
#' res$spearman           # -1 (perfectly miscalibrated)
predpol_calibration_audit <- function(areas, mean_risk, outcome_rate,
                                      group) {
  n <- length(areas)
  if (!(n == length(mean_risk) && n == length(outcome_rate) &&
        n == length(group))) {
    stop("areas, mean_risk, outcome_rate and group must all align",
         call. = FALSE)
  }
  if (n < 2L) {
    stop("need at least two areas to compare rankings", call. = FALSE)
  }
  areas <- as.character(areas)
  mean_risk <- as.numeric(mean_risk)
  outcome_rate <- as.numeric(outcome_rate)
  group <- as.character(group)

  warnings <- character(0)
  finite <- is.finite(mean_risk) & is.finite(outcome_rate)
  if (!all(finite)) {
    warnings <- c(warnings, sprintf(
      "%d area(s) had a non-finite risk or outcome value and were dropped.",
      sum(!finite)))
    areas <- areas[finite]; mean_risk <- mean_risk[finite]
    outcome_rate <- outcome_rate[finite]; group <- group[finite]
    n <- length(areas)
    if (n < 2L) {
      stop("fewer than two areas remain after dropping non-finite rows",
           call. = FALSE)
    }
  }

  # rank 1 = highest; rank() default ties.method is "average".
  risk_rank <- rank(-mean_risk)
  outcome_rank <- rank(-outcome_rate)
  rank_gap <- outcome_rank - risk_rank

  ct <- suppressWarnings(
    stats::cor.test(mean_risk, outcome_rate, method = "spearman"))
  rho <- unname(ct$estimate)
  pval <- ct$p.value

  per_group <- list(); group_n <- list()
  for (gv in unique(group)) {
    m <- group == gv
    per_group[[gv]] <- mean(rank_gap[m])
    group_n[[gv]] <- sum(m)
  }
  pg <- unlist(per_group)
  worst_group <- .frns_worst_abs_named(pg)
  worst <- pg[[worst_group]]

  cal <- if (rho >= 0.7) {
    sprintf(paste0("Overall the ranking is well calibrated (Spearman ",
                   "rho = %.2f)."), rho)
  } else if (rho >= 0.3) {
    sprintf("Overall calibration is weak (Spearman rho = %.2f).", rho)
  } else {
    sprintf("Overall the ranking is miscalibrated (Spearman rho = %.2f).",
            rho)
  }
  disp <- if (abs(worst) <= 0.5) {
    "No group's areas are systematically mis-ranked."
  } else if (worst > 0) {
    sprintf(paste0("Group '%s' is over-predicted: its areas are ranked, ",
                   "on average, %.1f rank positions more dangerous than ",
                   "their realised outcomes warrant."), worst_group, worst)
  } else {
    sprintf(paste0("Group '%s' is under-predicted: its areas are ranked, ",
                   "on average, %.1f rank positions less dangerous than ",
                   "their realised outcomes."), worst_group, abs(worst))
  }

  list(
    value = worst,
    spearman = rho,
    spearman_pvalue = pval,
    group_rank_gap = per_group,
    worst_group = worst_group,
    rank_gap = stats::setNames(as.list(rank_gap), areas),
    warnings = warnings,
    interpretation = paste(cal, disp)
  )
}


#' Descriptive disparity in a risk score across groups
#'
#' Reports per-group n / mean / median / sd, a one-way ANOVA for
#' whether group membership relates to the score, and each group's
#' mean-score gap from a reference group. A significant gap is not
#' itself proof of bias; pair this with `predpol_calibration_audit()`.
#'
#' @param score Continuous risk score, one per individual.
#' @param group Protected attribute, one per individual.
#' @param reference Reference group for the gaps; defaults to the
#'   lowest-scoring group.
#' @return A named list: `value` (mean-score spread), `spread`,
#'   `group_means`, `gaps`, `anova_f`, `anova_pvalue`, `significant`,
#'   `reference`, `warnings`, `interpretation`.
#' @export
#' @examples
#' res <- predpol_score_disparity(
#'   score = c(9, 10, 11, 19, 20, 21),
#'   group = c("A", "A", "A", "B", "B", "B"))
#' res$value         # 10  (group means 10 and 20)
#' res$significant   # TRUE
predpol_score_disparity <- function(score, group, reference = NULL) {
  if (length(score) != length(group)) {
    stop("score and group must be the same length", call. = FALSE)
  }
  score <- as.numeric(score)
  group <- as.character(group)
  if (length(unique(group)) < 2L) {
    stop("need at least two groups to measure disparity", call. = FALSE)
  }
  warnings <- character(0)
  finite <- is.finite(score)
  if (!all(finite)) {
    warnings <- c(warnings, sprintf(
      "%d non-finite score value(s) dropped.", sum(!finite)))
    score <- score[finite]; group <- group[finite]
  }
  groups <- unique(group)
  if (length(groups) < 2L) {
    stop("fewer than two groups remain after dropping NaNs", call. = FALSE)
  }

  means <- vapply(groups, function(g) mean(score[group == g]), numeric(1))
  names(means) <- groups
  per_group <- lapply(groups, function(g) {
    gv <- score[group == g]
    list(n = length(gv), mean = mean(gv), median = stats::median(gv),
         sd = if (length(gv) > 1L) stats::sd(gv) else NA_real_)
  })
  names(per_group) <- groups

  ow <- tryCatch(
    stats::oneway.test(score ~ factor(group), var.equal = TRUE),
    error = function(e) NULL)
  if (is.null(ow)) {
    fstat <- NA_real_; pval <- NA_real_
    warnings <- c(warnings, "ANOVA could not be computed.")
  } else {
    fstat <- unname(ow$statistic); pval <- ow$p.value
  }

  ref <- if (is.null(reference)) names(means)[which.min(means)]
         else as.character(reference)
  if (!ref %in% names(means)) {
    stop(sprintf("reference group '%s' not found", ref), call. = FALSE)
  }
  gaps <- as.list(means - means[[ref]])
  spread <- max(means) - min(means)
  significant <- isTRUE(is.finite(pval) && pval < 0.05)

  anova_line <- if (is.finite(pval)) {
    sprintf(paste0("A one-way ANOVA finds the between-group difference %s ",
                   "(F = %.2f, p = %.4f). "),
            if (significant) "statistically significant"
            else "not significant", fstat, pval)
  } else ""

  interp <- paste0(
    sprintf("Group mean risk scores span %.2f points (reference '%s'). ",
            spread, ref),
    anova_line,
    "Note: a score gap is not itself evidence of bias; pair this with ",
    "predpol_calibration_audit(), which compares the score against ",
    "realised outcomes.")

  list(
    value = spread,
    spread = spread,
    group_means = as.list(means),
    gaps = gaps,
    anova_f = fstat,
    anova_pvalue = pval,
    significant = significant,
    reference = ref,
    per_group = per_group,
    warnings = warnings,
    interpretation = interp
  )
}
