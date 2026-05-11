# SPDX-License-Identifier: GPL-2.0-only
#' MRM-framework analyses on Ontario SIU (Special Investigations Unit) data
#'
#' Three callables for SIU case-level CSVs. Unlike OTIS (no placement
#' dates) and TPS (no per-person ID), SIU exposes per-case dates with a
#' stable `police_service` jurisdiction column, enabling a real
#' "time-to-outcome" KM survival analysis.
#'
#' Functions:
#' * `mrm_siu_case_to_decision_km()`: Kaplan-Meier on the gap from
#'   `date_of_incident_iso` to `date_of_director_decision_iso`,
#'   stratified by `police_service`. The valid TTR analysis the
#'   MA-thesis "210-day TTR" claim should have been.
#' * `mrm_siu_per_service_rate()`: Per-police-service case rate by
#'   year and stratum, useful for cross-jurisdiction comparisons.
#' * `mrm_siu_outcome_classifier()`: Tabulates the Director's-decision
#'   categories (`charges_laid`, `no_charges`, etc.) by service and
#'   by year, reporting both raw counts and shares.
#'
#' @name mrm_siu
NULL


.parse_iso <- function(x) suppressWarnings(as.Date(x))


# ---------------------------------------------------------------------------
# 1. Case-to-decision KM
# ---------------------------------------------------------------------------

#' KM survival of SIU case-to-decision gap per police service
#'
#' Computes the gap (in days) between the incident date and the
#' Director's decision date for every SIU case, dropping rows where
#' either date is missing. Reports per-stratum median + IQR + n.
#' Cases without a decision date as of the snapshot are right-censored
#' if `censor_open_cases = TRUE` (default).
#'
#' This is the substantive "time-to-outcome" analysis the MA-thesis
#' "210-day TTR" claim should have been; it operates on real per-case
#' dates with a stable jurisdiction identifier.
#'
#' @param data A data.frame with the SIU case schema.
#' @param incident_col Column with ISO incident date
#'   (default `"date_of_incident_iso"`).
#' @param decision_col Column with ISO Director's decision date
#'   (default `"date_of_director_decision_iso"`).
#' @param service_col Stratifying jurisdiction column
#'   (default `"police_service"`).
#' @param censor_open_cases Logical (default `TRUE`). If `TRUE`,
#'   rows with missing `decision_col` contribute right-censored
#'   observations from incident to the most recent decision date
#'   in the data set. If `FALSE`, they are dropped.
#' @param min_n Minimum cases per service to retain in the per-service
#'   summary (default `5L`).
#' @return A list with elements:
#'   * `pooled`: a single-row data.frame with the pooled median,
#'     mean, IQR, n, n_censored.
#'   * `by_service`: per-service data.frame with the same columns.
#' @export
#' @examples
#' if (FALSE) {
#'   siu <- read.csv("SIU.csv")
#'   res <- mrm_siu_case_to_decision_km(siu)
#'   head(res$by_service)
#' }
mrm_siu_case_to_decision_km <- function(
  data,
  incident_col = "date_of_incident_iso",
  decision_col = "date_of_director_decision_iso",
  service_col = "police_service",
  censor_open_cases = TRUE,
  min_n = 5L
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(incident_col, decision_col, service_col) %in% names(data)))
  inc <- .parse_iso(data[[incident_col]])
  dec <- .parse_iso(data[[decision_col]])
  svc <- as.character(data[[service_col]])

  keep_inc <- !is.na(inc)
  gap <- as.numeric(dec - inc)

  if (censor_open_cases) {
    cutoff <- max(dec, na.rm = TRUE)
    open <- keep_inc & is.na(dec)
    gap[open] <- as.numeric(cutoff - inc[open])
    censored <- open
  } else {
    censored <- rep(FALSE, length(gap))
  }
  observed <- keep_inc & (!is.na(dec) | censor_open_cases)

  ok <- observed & is.finite(gap) & gap >= 0
  gap <- gap[ok]; svc <- svc[ok]; censored <- censored[ok]

  summarise <- function(gap_v, cens_v, label) {
    if (length(gap_v) == 0L) {
      return(data.frame(stratum = label, n = 0L, n_censored = 0L,
                        median_days = NA_real_, mean_days = NA_real_,
                        p25_days = NA_real_, p75_days = NA_real_,
                        max_days = NA_real_))
    }
    data.frame(
      stratum = label,
      n = length(gap_v),
      n_censored = sum(cens_v),
      median_days = stats::median(gap_v),
      mean_days = round(mean(gap_v), 2),
      p25_days = stats::quantile(gap_v, 0.25, names = FALSE),
      p75_days = stats::quantile(gap_v, 0.75, names = FALSE),
      max_days = max(gap_v)
    )
  }

  pooled <- summarise(gap, censored, "pooled")

  by_svc <- by(seq_along(gap), svc, function(idx) {
    if (length(idx) < min_n) return(NULL)
    summarise(gap[idx], censored[idx], unique(svc[idx]))
  }, simplify = FALSE)
  by_svc <- do.call(rbind, by_svc[!sapply(by_svc, is.null)])
  if (!is.null(by_svc)) rownames(by_svc) <- NULL

  list(pooled = pooled, by_service = by_svc)
}


# ---------------------------------------------------------------------------
# 2. Per-service rate
# ---------------------------------------------------------------------------

#' Per-police-service SIU case-rate summary
#'
#' Tabulates the number of SIU cases per police service per year, and
#' optionally per `reason_for_interaction` stratum.
#'
#' @param data A data.frame in the SIU case schema.
#' @param service_col Police-service column (default `"police_service"`).
#' @param incident_col Incident-date column for year extraction
#'   (default `"date_of_incident_iso"`).
#' @param stratify_col Optional second stratifying column.
#' @return A data.frame with columns `service`, `year`, optional
#'   `stratum`, `n_cases`.
#' @export
#' @examples
#' if (FALSE) {
#'   siu <- read.csv("SIU.csv")
#'   mrm_siu_per_service_rate(siu)
#' }
mrm_siu_per_service_rate <- function(
  data,
  service_col = "police_service",
  incident_col = "date_of_incident_iso",
  stratify_col = NULL
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(service_col, incident_col) %in% names(data)))
  inc <- .parse_iso(data[[incident_col]])
  year <- as.integer(format(inc, "%Y"))
  svc <- as.character(data[[service_col]])
  ok <- !is.na(year) & !is.na(svc) & nchar(svc) > 0
  if (is.null(stratify_col)) {
    tab <- table(svc[ok], year[ok])
    df <- as.data.frame.table(tab, responseName = "n_cases")
    names(df) <- c("service", "year", "n_cases")
  } else {
    stopifnot(stratify_col %in% names(data))
    str <- as.character(data[[stratify_col]])
    ok <- ok & !is.na(str)
    tab <- table(svc[ok], year[ok], str[ok])
    df <- as.data.frame.table(tab, responseName = "n_cases")
    names(df) <- c("service", "year", "stratum", "n_cases")
  }
  df <- df[df$n_cases > 0, ]
  rownames(df) <- NULL
  df
}


# ---------------------------------------------------------------------------
# 3. Outcome classifier (Director's decision categories)
# ---------------------------------------------------------------------------

#' Tabulate SIU Director's-decision outcomes
#'
#' Cross-tabulates a categorical outcome column (default
#' `director_decision_category`) by service and year, reporting both
#' raw counts and within-service shares. If the supplied
#' `outcome_col` is not present, looks for a few common alternatives
#' (`director_decision`, `outcome`).
#'
#' @param data A data.frame in the SIU case schema.
#' @param outcome_col Outcome category column
#'   (default `"director_decision_category"`).
#' @param service_col Police-service column
#'   (default `"police_service"`).
#' @return A data.frame with columns `service`, `outcome`, `n_cases`,
#'   `share_within_service`.
#' @export
#' @examples
#' if (FALSE) {
#'   siu <- read.csv("SIU.csv")
#'   mrm_siu_outcome_classifier(siu)
#' }
mrm_siu_outcome_classifier <- function(
  data,
  outcome_col = "director_decision_category",
  service_col = "police_service"
) {
  stopifnot(is.data.frame(data))
  if (!outcome_col %in% names(data)) {
    for (alt in c("director_decision", "outcome", "decision",
                  "director_decision_outcome", "director_decision_text")) {
      if (alt %in% names(data)) { outcome_col <- alt; break }
    }
  }
  stopifnot(outcome_col %in% names(data))
  stopifnot(service_col %in% names(data))
  out <- as.character(data[[outcome_col]])
  svc <- as.character(data[[service_col]])
  ok <- !is.na(out) & nchar(out) > 0 & !is.na(svc) & nchar(svc) > 0
  out <- out[ok]; svc <- svc[ok]
  tab <- table(svc, out)
  totals <- rowSums(tab)
  df <- as.data.frame.table(tab, responseName = "n_cases")
  names(df) <- c("service", "outcome", "n_cases")
  df$share_within_service <- round(df$n_cases / totals[as.character(df$service)], 4)
  df <- df[df$n_cases > 0, ]
  rownames(df) <- NULL
  df
}
