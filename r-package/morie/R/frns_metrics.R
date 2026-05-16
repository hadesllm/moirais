# SPDX-License-Identifier: AGPL-3.0-or-later
#' Group-disparity metrics for auditing classification and risk systems
#'
#' R parity for the Python `morie.fairness.metrics` module. Every
#' callable here is an *audit* measure: given the decisions a system
#' made (and, where available, the realised ground truth) plus a
#' protected attribute such as race, it quantifies whether outcomes
#' differ across groups. None of these functions make predictions; they
#' only measure disparity in predictions that already exist.
#'
#' Functions:
#' * `fairness_disparate_impact()`: the EEOC four-fifths rule.
#' * `fairness_demographic_parity()`: favourable-rate gap.
#' * `fairness_equalized_odds()`: TPR/FPR gaps (needs ground truth).
#' * `fairness_average_odds_difference()`: mean TPR+FPR gap.
#' * `fairness_gini()`: concentration of a score distribution.
#' * `fairness_bias_amplification()`: composite `Delta_parity * G`.
#'
#' Each returns a named `list` with the metric value, a per-group
#' breakdown, any advisory `warnings`, and a plain-language
#' `interpretation`, mirroring the payload of the Python
#' `RichResult`.
#'
#' Prior art reimplemented independently (no code copied): IBM AI
#' Fairness 360 metric definitions; the COMPAS audit in pbiecek's
#' *XAI Stories*; the SciencesPo *Predictive-policing-Chicago* project
#' (Lacherade, Szabo, Krikava & Aeby, 2021); and Barman & Barman,
#' arXiv:2603.18987 (the Bias Amplification Score).
#'
#' @name frns_metrics
NULL

.FRNS_FOUR_FIFTHS <- 0.8  # EEOC four-fifths adverse-impact threshold


# ---- internal helpers -----------------------------------------------------

.frns_check_aligned <- function(...) {
  args <- list(...)
  lengths <- vapply(args, function(a) length(a[[2]]), integer(1))
  if (length(unique(lengths)) > 1L) {
    nm <- vapply(args, function(a) a[[1]], character(1))
    stop(sprintf("length mismatch across inputs: %s",
                 paste(sprintf("%s=%d", nm, lengths), collapse = ", ")),
         call. = FALSE)
  }
  if (lengths[1] == 0L) stop("inputs are empty", call. = FALSE)
}

.frns_favorable_rates <- function(outcome, group, favorable) {
  groups <- unique(group)
  rates <- list()
  for (g in groups) {
    mask <- group == g
    n <- sum(mask)
    rate <- if (n > 0L) mean(outcome[mask] == favorable) else NA_real_
    rates[[as.character(g)]] <- list(value = g, n = n, rate = rate)
  }
  rates
}

.frns_resolve_privileged <- function(privileged, rates) {
  # Returns list(privileged = <key>, warning = <chr or NULL>).
  keys <- names(rates)
  if (!is.null(privileged)) {
    pk <- as.character(privileged)
    if (!pk %in% keys) {
      stop(sprintf("privileged group '%s' not found; groups present: %s",
                   pk, paste(keys, collapse = ", ")), call. = FALSE)
    }
    return(list(privileged = pk, warning = NULL))
  }
  rate_vals <- vapply(rates, function(r) r$rate, numeric(1))
  pk <- keys[which.max(rate_vals)]
  list(privileged = pk, warning = sprintf(
    paste0("`privileged` not given; inferred as '%s' (the group with the ",
           "highest favourable-outcome rate). Pass `privileged=` to audit ",
           "against a specific reference group."), pk))
}

.frns_rates_from_labels <- function(y_true, y_pred, group, favorable) {
  groups <- unique(group)
  out <- list()
  for (g in groups) {
    m <- group == g
    gt <- y_true[m]; gp <- y_pred[m]
    pos <- gt == favorable
    neg <- !pos
    tpr <- if (any(pos)) mean(gp[pos] == favorable) else NA_real_
    fpr <- if (any(neg)) mean(gp[neg] == favorable) else NA_real_
    out[[as.character(g)]] <- list(value = g, n = sum(m),
                                   tpr = tpr, fpr = fpr)
  }
  out
}

.frns_gini <- function(x) {
  # Gini via the sorted-rank formula; equals sum_i sum_j |x_i-x_j| /
  # (2 n sum x). Returns 0 for all-zero or single-element input.
  x <- sort(as.numeric(x))
  n <- length(x)
  total <- sum(x)
  if (n < 2L || total <= 0) return(0)
  idx <- seq_len(n)
  (2 * sum(idx * x)) / (n * total) - (n + 1) / n
}

.frns_worst_abs <- function(values) {
  # The element with the largest absolute value (finite only); NA if none.
  finite <- values[is.finite(values)]
  if (length(finite) == 0L) return(NA_real_)
  finite[which.max(abs(finite))]
}


# ---- 1. disparate impact --------------------------------------------------

#' Disparate Impact Ratio (EEOC four-fifths rule)
#'
#' For each group, the disparate-impact ratio is its favourable-outcome
#' rate divided by the privileged group's rate. A value below 0.8 is the
#' standard legal indicator of adverse impact.
#'
#' @param y_pred Vector of decisions/assignments, one per individual.
#' @param group Vector of protected-attribute values (e.g. race).
#' @param privileged The reference group. If `NULL` (default) the
#'   highest-rate group is used and a warning is attached.
#' @param favorable The value of `y_pred` counted as the favourable
#'   outcome (default `1`).
#' @return A named list: `value` (worst ratio), `ratios`, `rates`,
#'   `privileged`, `adverse_impact`, `threshold`, `warnings`,
#'   `interpretation`.
#' @export
#' @examples
#' pred <- c(1, 1, 1, 1, 1, 1, 1, 1, 0, 0)
#' race <- c(rep("A", 5), rep("B", 5))
#' res <- fairness_disparate_impact(pred, race, privileged = "A")
#' res$value           # 0.6  (group B rate 0.6 / group A rate 1.0)
#' res$adverse_impact  # TRUE
fairness_disparate_impact <- function(y_pred, group, privileged = NULL,
                                      favorable = 1) {
  .frns_check_aligned(list("y_pred", y_pred), list("group", group))
  rates <- .frns_favorable_rates(y_pred, group, favorable)
  if (length(rates) < 2L) {
    stop("need at least two groups to measure disparity", call. = FALSE)
  }
  warnings <- character(0)
  pr <- .frns_resolve_privileged(privileged, rates)
  priv <- pr$privileged
  if (!is.null(pr$warning)) warnings <- c(warnings, pr$warning)
  base <- rates[[priv]]$rate
  if (base == 0) {
    warnings <- c(warnings, sprintf(
      paste0("privileged group '%s' has a zero favourable-outcome rate; ",
             "disparate-impact ratios are undefined and reported as NA."),
      priv))
  }

  ratios <- list()
  for (k in names(rates)) {
    ratios[[k]] <- if (k == priv) 1.0
                   else if (base == 0) NA_real_
                   else rates[[k]]$rate / base
  }
  non_ref <- unlist(ratios[names(ratios) != priv])
  finite <- non_ref[is.finite(non_ref)]
  worst <- if (length(finite)) min(finite) else NA_real_
  adverse <- isTRUE(is.finite(worst) && worst < .FRNS_FOUR_FIFTHS)

  interp <- if (!is.finite(worst)) {
    "Disparate-impact ratio could not be computed (privileged group has no favourable outcomes)."
  } else if (adverse) {
    sprintf(paste0("Adverse impact detected: the worst disparate-impact ",
                   "ratio is %.3f, below the 0.80 four-fifths threshold."),
            worst)
  } else {
    sprintf(paste0("No adverse impact under the four-fifths rule: the ",
                   "worst disparate-impact ratio is %.3f (>= 0.80)."), worst)
  }

  list(
    value = worst,
    ratios = ratios,
    rates = lapply(rates, function(r) r$rate),
    privileged = priv,
    adverse_impact = adverse,
    threshold = .FRNS_FOUR_FIFTHS,
    warnings = warnings,
    interpretation = interp
  )
}


# ---- 2. demographic parity ------------------------------------------------

#' Demographic Parity Gap
#'
#' The additive difference in favourable-outcome rates,
#' `rate(group) - rate(privileged)`. Demographic parity holds when every
#' group receives favourable outcomes at the same rate.
#'
#' @inheritParams fairness_disparate_impact
#' @return A named list: `value` (largest absolute gap), `gaps`, `rates`,
#'   `privileged`, `warnings`, `interpretation`.
#' @export
#' @examples
#' pred <- c(1, 1, 1, 1, 0, 0, 0, 1, 0, 0)
#' race <- c(rep("A", 5), rep("B", 5))
#' res <- fairness_demographic_parity(pred, race, privileged = "A")
#' res$value   # -0.6  (group B rate 0.2 minus group A rate 0.8)
fairness_demographic_parity <- function(y_pred, group, privileged = NULL,
                                        favorable = 1) {
  .frns_check_aligned(list("y_pred", y_pred), list("group", group))
  rates <- .frns_favorable_rates(y_pred, group, favorable)
  if (length(rates) < 2L) {
    stop("need at least two groups to measure disparity", call. = FALSE)
  }
  warnings <- character(0)
  pr <- .frns_resolve_privileged(privileged, rates)
  priv <- pr$privileged
  if (!is.null(pr$warning)) warnings <- c(warnings, pr$warning)
  base <- rates[[priv]]$rate

  gaps <- list()
  for (k in names(rates)) gaps[[k]] <- rates[[k]]$rate - base
  non_ref <- unlist(gaps[names(gaps) != priv])
  worst <- .frns_worst_abs(non_ref)

  interp <- sprintf(
    paste0("The largest favourable-rate gap is %+.3f (group rate minus ",
           "the '%s' reference rate). %s"),
    worst, priv,
    if (is.finite(worst) && abs(worst) >= 0.1) {
      "Favourable-outcome rates differ materially across groups."
    } else {
      "Favourable-outcome rates are close to parity."
    })

  list(
    value = worst,
    gaps = gaps,
    rates = lapply(rates, function(r) r$rate),
    privileged = priv,
    warnings = warnings,
    interpretation = interp
  )
}


# ---- 3. equalized odds ----------------------------------------------------

#' Equalized Odds (true- and false-positive-rate gaps)
#'
#' Equalized odds holds when the true-positive rate (TPR) and
#' false-positive rate (FPR) are equal across groups. Needs ground-truth
#' labels, so it audits a system's *errors*, not just its decision
#' rates.
#'
#' @param y_true Vector of realised ground-truth outcomes.
#' @param y_pred Vector of system decisions.
#' @param group Vector of protected-attribute values.
#' @param privileged The reference group (inferred if `NULL`).
#' @param favorable The value treated as the positive class (default `1`).
#' @return A named list: `value` (largest absolute TPR/FPR gap),
#'   `tpr_gaps`, `fpr_gaps`, `rates`, `privileged`, `violation`,
#'   `warnings`, `interpretation`.
#' @export
#' @examples
#' truth <- c(1, 0, 1, 0, 1, 0, 1, 0)
#' pred  <- c(1, 0, 1, 0, 1, 1, 0, 1)
#' race  <- c(rep("A", 4), rep("B", 4))
#' res <- fairness_equalized_odds(truth, pred, race, privileged = "A")
#' res$violation   # TRUE
fairness_equalized_odds <- function(y_true, y_pred, group,
                                    privileged = NULL, favorable = 1) {
  .frns_check_aligned(list("y_true", y_true), list("y_pred", y_pred),
                      list("group", group))
  per <- .frns_rates_from_labels(y_true, y_pred, group, favorable)
  if (length(per) < 2L) {
    stop("need at least two groups to measure disparity", call. = FALSE)
  }
  warnings <- character(0)
  rate_view <- lapply(per, function(d) list(value = d$value, n = d$n,
                                            rate = d$tpr))
  pr <- .frns_resolve_privileged(privileged, rate_view)
  priv <- pr$privileged
  if (!is.null(pr$warning)) warnings <- c(warnings, pr$warning)
  base_tpr <- per[[priv]]$tpr
  base_fpr <- per[[priv]]$fpr

  tpr_gaps <- list(); fpr_gaps <- list()
  for (k in names(per)) {
    tpr_gaps[[k]] <- per[[k]]$tpr - base_tpr
    fpr_gaps[[k]] <- per[[k]]$fpr - base_fpr
    if (is.na(per[[k]]$tpr) || is.na(per[[k]]$fpr)) {
      warnings <- c(warnings, sprintf(
        paste0("group '%s' has no positive or no negative ground-truth ",
               "cases; its TPR/FPR (and gaps) are partly undefined."), k))
    }
  }
  all_gaps <- c(unlist(tpr_gaps[names(tpr_gaps) != priv]),
                unlist(fpr_gaps[names(fpr_gaps) != priv]))
  worst <- .frns_worst_abs(all_gaps)
  violation <- isTRUE(is.finite(worst) && abs(worst) >= 0.1)

  interp <- sprintf(
    "The largest equalized-odds gap is %+.3f. %s", worst,
    if (violation) {
      "Error rates differ substantially across groups."
    } else {
      "TPR and FPR are close across groups."
    })

  list(
    value = worst,
    tpr_gaps = tpr_gaps,
    fpr_gaps = fpr_gaps,
    rates = per,
    privileged = priv,
    violation = violation,
    warnings = warnings,
    interpretation = interp
  )
}


# ---- 4. average odds difference -------------------------------------------

#' Average Odds Difference
#'
#' For each non-reference group,
#' `0.5 * ((FPR_group - FPR_ref) + (TPR_group - TPR_ref))`. Zero means
#' parity of errors. This is the single-number summary used in IBM
#' AIF360 and the COMPAS *XAI Stories* audit.
#'
#' @inheritParams fairness_equalized_odds
#' @return A named list: `value` (largest absolute AOD),
#'   `average_odds_difference`, `rates`, `privileged`, `warnings`,
#'   `interpretation`.
#' @export
#' @examples
#' truth <- c(1, 0, 1, 0, 1, 0, 1, 0)
#' pred  <- c(1, 0, 1, 0, 1, 1, 0, 1)
#' race  <- c(rep("A", 4), rep("B", 4))
#' res <- fairness_average_odds_difference(truth, pred, race,
#'                                         privileged = "A")
#' res$value   # 0.25
fairness_average_odds_difference <- function(y_true, y_pred, group,
                                             privileged = NULL,
                                             favorable = 1) {
  .frns_check_aligned(list("y_true", y_true), list("y_pred", y_pred),
                      list("group", group))
  per <- .frns_rates_from_labels(y_true, y_pred, group, favorable)
  if (length(per) < 2L) {
    stop("need at least two groups to measure disparity", call. = FALSE)
  }
  warnings <- character(0)
  rate_view <- lapply(per, function(d) list(value = d$value, n = d$n,
                                            rate = d$tpr))
  pr <- .frns_resolve_privileged(privileged, rate_view)
  priv <- pr$privileged
  if (!is.null(pr$warning)) warnings <- c(warnings, pr$warning)
  base_tpr <- per[[priv]]$tpr
  base_fpr <- per[[priv]]$fpr

  aod <- list()
  for (k in names(per)) {
    aod[[k]] <- 0.5 * ((per[[k]]$fpr - base_fpr) +
                       (per[[k]]$tpr - base_tpr))
  }
  non_ref <- unlist(aod[names(aod) != priv])
  worst <- .frns_worst_abs(non_ref)

  interp <- sprintf(
    paste0("The largest average odds difference is %+.3f. Zero is parity; ",
           "values away from zero mean the combined error profile favours ",
           "one group over another."), worst)

  list(
    value = worst,
    average_odds_difference = aod,
    rates = per,
    privileged = priv,
    warnings = warnings,
    interpretation = interp
  )
}


# ---- 5. Gini --------------------------------------------------------------

#' Gini Coefficient (concentration / inequality of a distribution)
#'
#' Ranges from 0 (perfect equality) to nearly 1 (one unit holds
#' everything). Applied to risk scores or patrol counts it measures how
#' unequally a system concentrates its attention. With `group` supplied,
#' a per-group Gini is also reported.
#'
#' @param values Vector of non-negative quantities.
#' @param group Optional vector of protected-attribute values; enables
#'   the per-group breakdown.
#' @return A named list: `value` (overall Gini), `gini`, `per_group`,
#'   `warnings`, `interpretation`.
#' @export
#' @examples
#' fairness_gini(c(5, 5, 5, 5))$value      # 0
#' fairness_gini(c(0, 0, 0, 100))$value    # 0.75
fairness_gini <- function(values, group = NULL) {
  if (length(values) == 0L) stop("values is empty", call. = FALSE)
  vals <- as.numeric(values)
  warnings <- character(0)
  if (any(vals < 0)) {
    warnings <- c(warnings, paste0(
      "negative values present; the Gini coefficient assumes ",
      "non-negative quantities and the result may be uninformative."))
  }
  overall <- .frns_gini(vals)

  per_group <- list()
  if (!is.null(group)) {
    .frns_check_aligned(list("values", vals), list("group", group))
    for (g in unique(group)) {
      per_group[[as.character(g)]] <- .frns_gini(vals[group == g])
    }
  }

  interp <- sprintf("Gini = %.3f. %s", overall,
    if (overall >= 0.5) {
      "The quantity is highly concentrated."
    } else {
      "The quantity is relatively evenly spread."
    })

  list(
    value = overall,
    gini = overall,
    per_group = per_group,
    warnings = warnings,
    interpretation = interp
  )
}


# ---- 6. bias amplification score ------------------------------------------

#' Bias Amplification Score (composite parity-gap times inequality)
#'
#' `BAS = Delta_parity * G`, where `Delta_parity` is the demographic
#' parity gap of the worst-affected group and `G` is the Gini
#' coefficient of the per-group favourable-outcome rates. Large only
#' when a directional disparity coincides with high overall inequality.
#'
#' Reimplemented from Barman & Barman, "Unmasking Algorithmic Bias in
#' Predictive Policing" (arXiv:2603.18987).
#'
#' @inheritParams fairness_disparate_impact
#' @return A named list: `value` (BAS), `bias_amplification_score`,
#'   `demographic_parity_gap`, `gini`, `rates`, `privileged`,
#'   `warnings`, `interpretation`.
#' @export
#' @examples
#' pred <- c(1, 1, 1, 1, 0, 0, 0, 0)
#' race <- c(rep("A", 4), rep("B", 4))
#' res <- fairness_bias_amplification(pred, race, privileged = "A")
#' res$value   # -0.5  (parity gap -1.0 times Gini 0.5)
fairness_bias_amplification <- function(y_pred, group, privileged = NULL,
                                        favorable = 1) {
  .frns_check_aligned(list("y_pred", y_pred), list("group", group))
  rates <- .frns_favorable_rates(y_pred, group, favorable)
  if (length(rates) < 2L) {
    stop("need at least two groups to measure disparity", call. = FALSE)
  }
  warnings <- character(0)
  pr <- .frns_resolve_privileged(privileged, rates)
  priv <- pr$privileged
  if (!is.null(pr$warning)) warnings <- c(warnings, pr$warning)
  base <- rates[[priv]]$rate

  gaps <- list()
  for (k in names(rates)) gaps[[k]] <- rates[[k]]$rate - base
  non_ref <- unlist(gaps[names(gaps) != priv])
  delta_parity <- .frns_worst_abs(non_ref)
  if (is.na(delta_parity)) delta_parity <- 0

  rate_vec <- vapply(rates, function(r) r$rate, numeric(1))
  gini <- .frns_gini(rate_vec)
  bas <- delta_parity * gini

  interp <- sprintf(
    paste0("Bias Amplification Score = %+.4f (parity gap %+.3f times ",
           "Gini %.3f). %s"), bas, delta_parity, gini,
    if (abs(bas) >= 0.05) {
      "Both a directional disparity and substantial inequality are present."
    } else {
      "At least one component is small, so little amplification is indicated."
    })

  list(
    value = bas,
    bias_amplification_score = bas,
    demographic_parity_gap = delta_parity,
    gini = gini,
    rates = lapply(rates, function(r) r$rate),
    privileged = priv,
    warnings = warnings,
    interpretation = interp
  )
}
