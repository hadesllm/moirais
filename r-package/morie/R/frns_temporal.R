# SPDX-License-Identifier: AGPL-3.0-or-later
#' Multi-city temporal disparity audit
#'
#' R parity for the Python `morie.fairness.temporal` module. The four
#' disparity metrics — Disparate Impact Ratio, Demographic Parity Gap,
#' Gini coefficient, and Bias Amplification Score — are computed for
#' each `(city, period)` cell and aggregated per city, so temporal
#' instability and cross-city divergence become visible.
#'
#' Reimplements the longitudinal, multi-city audit of Barman & Barman,
#' arXiv:2603.18987. Its central lesson: bias metrics are not stable
#' from one deployment cycle to the next and must be recomputed per
#' period and per city.
#'
#' @name frns_temporal
NULL


#' Audit how disparity metrics move over time and across cities
#'
#' For every `(city, period)` cell the four disparity metrics are
#' computed; per city the audit then reports the mean of each metric,
#' the count of periods with DIR above 1, and the DIR temporal range
#' (max minus min) — the headline measure of instability.
#'
#' @param period Time-period label for each record (e.g. `"2019-03"`).
#' @param city City label for each record.
#' @param y_pred The decision/assignment for each record.
#' @param group Protected attribute for each record.
#' @param privileged Reference group; inferred globally from the pooled
#'   data when `NULL` so every cell uses the same reference.
#' @param favorable Value of `y_pred` counted as favourable (default `1`).
#' @return A named list: `value` (worst per-city DIR range),
#'   `worst_dir_range`, `cross_city_dir_spread`, `per_city`, `cells`,
#'   `privileged`, `warnings`, `interpretation`.
#' @export
#' @examples
#' period <- c(rep("p1", 10), rep("p2", 10))
#' city <- rep("A", 20)
#' pred <- rep(c(1, 1, 1, 1, 1, 1, 1, 1, 0, 0), 2)
#' grp <- rep(c(rep("X", 5), rep("Y", 5)), 2)
#' res <- predpol_temporal_audit(period, city, pred, grp, privileged = "X")
#' res$per_city$A$dir_range   # 0 — disparity is stable across periods
predpol_temporal_audit <- function(period, city, y_pred, group,
                                   privileged = NULL, favorable = 1) {
  n <- length(period)
  if (!(n == length(city) && n == length(y_pred) && n == length(group))) {
    stop("period, city, y_pred and group must all align", call. = FALSE)
  }
  if (n == 0L) stop("inputs are empty", call. = FALSE)
  period <- as.character(period)
  city <- as.character(city)
  group <- as.character(group)

  warnings <- character(0)
  if (is.null(privileged)) {
    gs <- unique(group)
    rates <- vapply(gs, function(g) mean(y_pred[group == g] == favorable),
                    numeric(1))
    privileged <- gs[which.max(rates)]
    warnings <- c(warnings, sprintf(
      paste0("`privileged` not given; inferred globally as '%s' so every ",
             "cell uses the same reference group."), privileged))
  } else {
    privileged <- as.character(privileged)
  }

  cells <- list()
  skipped <- 0L
  for (cc in unique(city)) {
    for (pp in sort(unique(period[city == cc]))) {
      m <- city == cc & period == pp
      cg <- group[m]
      cy <- y_pred[m]
      cgu <- unique(cg)
      if (length(cgu) < 2L || !(privileged %in% cgu)) {
        skipped <- skipped + 1L
        next
      }
      di <- fairness_disparate_impact(
        cy, cg, privileged = privileged, favorable = favorable)$value
      pg <- fairness_demographic_parity(
        cy, cg, privileged = privileged, favorable = favorable)$value
      rate_vec <- vapply(cgu, function(g) mean(cy[cg == g] == favorable),
                         numeric(1))
      gini <- fairness_gini(rate_vec)$value
      bas <- fairness_bias_amplification(
        cy, cg, privileged = privileged, favorable = favorable)$value
      cells[[length(cells) + 1L]] <- list(
        city = cc, period = pp, n = sum(m),
        dir = di, parity_gap = pg, gini = gini, bas = bas)
    }
  }
  if (skipped > 0L) {
    warnings <- c(warnings, sprintf(
      "%d (city, period) cell(s) were skipped (fewer than two groups, or the privileged group absent).",
      skipped))
  }
  if (length(cells) == 0L) {
    stop("no (city, period) cell had enough groups to audit", call. = FALSE)
  }

  cities <- unique(vapply(cells, function(x) x$city, character(1)))
  per_city <- list()
  for (cc in cities) {
    sub <- Filter(function(x) x$city == cc, cells)
    dirs <- vapply(sub, function(x) x$dir, numeric(1))
    dirs_f <- dirs[is.finite(dirs)]
    per_city[[cc]] <- list(
      n_periods = length(sub),
      mean_dir = mean(dirs_f),
      mean_parity_gap = mean(vapply(sub, function(x) x$parity_gap, numeric(1))),
      mean_gini = mean(vapply(sub, function(x) x$gini, numeric(1))),
      mean_bas = mean(vapply(sub, function(x) x$bas, numeric(1))),
      dir_min = if (length(dirs_f)) min(dirs_f) else NA_real_,
      dir_max = if (length(dirs_f)) max(dirs_f) else NA_real_,
      dir_range = if (length(dirs_f)) max(dirs_f) - min(dirs_f) else NA_real_,
      periods_dir_gt1 = sum(dirs_f > 1.0)
    )
  }

  ranges <- vapply(per_city, function(v) v$dir_range, numeric(1))
  ranges_f <- ranges[is.finite(ranges)]
  worst_range <- if (length(ranges_f)) max(ranges_f) else NA_real_
  mean_dirs <- vapply(per_city, function(v) v$mean_dir, numeric(1))
  mean_dirs_f <- mean_dirs[is.finite(mean_dirs)]
  cross <- if (length(mean_dirs_f) >= 2L) {
    max(mean_dirs_f) - min(mean_dirs_f)
  } else 0

  stab <- if (is.finite(worst_range) && worst_range >= 0.5) {
    sprintf(paste0("Bias is temporally unstable: the Disparate Impact ",
                   "Ratio swings by up to %.3f across periods within a ",
                   "single city; the metric must be recomputed every ",
                   "period."), worst_range)
  } else {
    "The Disparate Impact Ratio is reasonably stable across periods."
  }
  div <- if (length(per_city) >= 2L && cross >= 0.3) {
    sprintf(paste0(" Bias also diverges across cities: mean annual DIR ",
                   "spans %.3f between cities."), cross)
  } else ""

  list(
    value = worst_range,
    worst_dir_range = worst_range,
    cross_city_dir_spread = cross,
    per_city = per_city,
    cells = cells,
    privileged = privileged,
    warnings = warnings,
    interpretation = paste0(stab, div)
  )
}
