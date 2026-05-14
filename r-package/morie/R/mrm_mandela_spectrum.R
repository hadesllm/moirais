# SPDX-License-Identifier: MIT OR Apache-2.0

#' Mandela Rules apples-to-apples spectrum on OTIS b01
#'
#' R parity of \code{morie.mrm_otis_mandela_spectrum()}.  Computes a
#' full grid of provincial Mandela-classified rates across four
#' denominator conventions x three meaningful-contact proxies, so the
#' \emph{Cross-jurisdiction comparison} table in the MRM formulations
#' paper (Section 5.3) can be reproduced from a single function call.
#'
#' Denominator conventions:
#' \describe{
#'   \item{\code{row}}{per-placement rate (b01 row count)}
#'   \item{\code{individual_any}}{share of within-year individuals with
#'     any placement satisfying the criterion}
#'   \item{\code{individual_cumulative}}{share of within-year
#'     individuals whose cumulative within-year segregation days
#'     exceeds the threshold}
#'   \item{\code{c11_aggregate}}{the duration-band aggregate from c11
#'     (requires \code{c11_data} to be supplied)}
#' }
#'
#' Meaningful-contact proxies (Rule 44, derived from b01 alert flags):
#' \describe{
#'   \item{\code{none}}{Rule 43 only; no contact proxy applied}
#'   \item{\code{any_alert}}{Rule 43 \\cap{} (any of MH/SR/SW alert
#'     active); these placements receive staff contact, so this is the
#'     looser contact-failure proxy}
#'   \item{\code{no_alert}}{Rule 43 \\cap{} (no alert active);
#'     strictest contact-failure proxy}
#' }
#'
#' @param data OTIS b01 data.frame.
#' @param duration_col,year_col,id_col Column names.
#' @param threshold_days Rule 43 duration threshold (UN: 15 days).
#' @param alert_cols Three b01 alert columns (\code{"Yes"} = active).
#' @param contact_proxies Subset of
#'   \code{c("none","any_alert","no_alert")}.
#' @param denominators Subset of \code{c("row","individual_any",
#'   "individual_cumulative","c11_aggregate")}.
#' @param c11_data Optional c11 aggregate frame for the
#'   \code{c11_aggregate} denominator.
#'
#' @return Tidy long-format data.frame with one row per
#'   (year, denominator, contact_proxy) cell, columns
#'   \code{year}, \code{denominator}, \code{contact_proxy},
#'   \code{n_eligible}, \code{n_mandela}, \code{rate}, \code{pct}.
#'
#' @references
#' United Nations General Assembly (2015). United Nations Standard
#' Minimum Rules for the Treatment of Prisoners (the Nelson Mandela
#' Rules). A/RES/70/175. Rule 43 = prolonged (>15 days). Rule 44 =
#' >=22 hours/day, no meaningful human contact.
#'
#' @export
#' @examples
#' if (FALSE) {
#'   b01 <- read.csv("b01_segregation_detailed_dataset.csv")
#'   spec <- mrm_otis_mandela_spectrum(b01)
#'   head(spec)
#' }
mrm_otis_mandela_spectrum <- function(
  data,
  duration_col = "NumberConsecutiveDays_Segregation",
  year_col     = "EndFiscalYear",
  id_col       = "UniqueIndividual_ID",
  threshold_days = 15L,
  alert_cols   = c("MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert"),
  contact_proxies = c("none", "any_alert", "no_alert"),
  denominators    = c("row", "individual_any", "individual_cumulative"),
  c11_data     = NULL
) {
  stopifnot(is.data.frame(data))
  stopifnot(all(c(duration_col, year_col, id_col) %in% names(data)))
  d <- data
  d[["_dur"]] <- suppressWarnings(as.numeric(d[[duration_col]]))
  d[["_long"]] <- !is.na(d[["_dur"]]) & d[["_dur"]] > threshold_days
  d[["_any_alert"]] <- FALSE
  d[["_no_alert"]] <- TRUE
  for (c in alert_cols) {
    if (!c %in% names(d)) next
    yes <- tolower(trimws(as.character(d[[c]]))) == "yes"
    d[["_any_alert"]] <- d[["_any_alert"]] | yes
    d[["_no_alert"]]  <- d[["_no_alert"]]  & !yes
  }

  years <- sort(unique(d[[year_col]]))
  rows  <- list()

  for (y in c(as.list(years), list("pooled"))) {
    if (identical(y, "pooled")) {
      ymask <- rep(TRUE, nrow(d))
      label <- "pooled"
    } else {
      ymask <- d[[year_col]] == y
      label <- as.character(y)
    }
    for (proxy in contact_proxies) {
      elig <- switch(proxy,
        "none"      = d[["_long"]],
        "any_alert" = d[["_long"]] & d[["_any_alert"]],
        "no_alert"  = d[["_long"]] & d[["_no_alert"]],
        stop(sprintf("unknown contact_proxy %s", proxy))
      ) & ymask

      for (denom in denominators) {
        if (denom == "row") {
          n_d <- sum(ymask)
          n_m <- sum(elig)
        } else if (denom == "individual_any") {
          ids   <- unique(stats::na.omit(d[[id_col]][ymask]))
          ids_m <- unique(stats::na.omit(d[[id_col]][elig]))
          n_d <- length(ids)
          n_m <- length(ids_m)
        } else if (denom == "individual_cumulative") {
          sub <- d[ymask, , drop = FALSE]
          cum <- tapply(sub[["_dur"]], sub[[id_col]], sum, na.rm = TRUE)
          n_d <- length(cum)
          if (proxy == "none") {
            n_m <- sum(cum > threshold_days, na.rm = TRUE)
          } else {
            ids_m <- unique(stats::na.omit(d[[id_col]][elig]))
            cum_long <- cum > threshold_days
            n_m <- sum(cum_long & (names(cum) %in% ids_m), na.rm = TRUE)
          }
        } else if (denom == "c11_aggregate") {
          if (is.null(c11_data)) next
          sub <- if (identical(y, "pooled")) c11_data else c11_data[c11_data[[year_col]] == y, , drop = FALSE]
          if (!"NumberIndividuals_Segregation" %in% names(sub)) next
          if (!"Aggregate_Duration" %in% names(sub)) next
          n_d <- sum(sub[["NumberIndividuals_Segregation"]])
          ab <- vapply(as.character(sub[["Aggregate_Duration"]]), function(b) {
            grepl("Greater than", b) | any(as.integer(regmatches(b, gregexpr("[0-9]+", b))[[1]]) > threshold_days)
          }, logical(1))
          n_m <- sum(sub[["NumberIndividuals_Segregation"]][ab])
        } else stop(sprintf("unknown denominator %s", denom))

        rate <- if (n_d > 0) n_m / n_d else NA_real_
        rows[[length(rows) + 1L]] <- data.frame(
          year = label, denominator = denom, contact_proxy = proxy,
          n_eligible = as.integer(n_d), n_mandela = as.integer(n_m),
          rate = round(rate, 6), pct = round(100 * rate, 2),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  do.call(rbind, rows)
}
