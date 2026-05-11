#' Mandela Rules classifier for solitary-confinement placements
#'
#' Classify placement records under the United Nations Standard Minimum
#' Rules for the Treatment of Prisoners (the Nelson Mandela Rules,
#' UN A/RES/70/175) which define prolonged solitary confinement as any
#' continuous placement exceeding fifteen days. Provides three
#' denominator conventions and an optional broader-restrictive-confinement
#' classification that adds high-alert placements to the numerator.
#'
#' The provincial classification operates on duration alone (the
#' `duration_col` column). The federal classification additionally
#' requires unmet "meaningful contact" criteria (Sprott-Doob 2023);
#' if `meaningful_contact_col` is supplied, that column is treated as a
#' 1-if-met indicator and rows with met-contact are excluded from the
#' numerator regardless of duration.
#'
#' @param data A data.frame or data.table containing at minimum the
#'   placement-duration column and a fiscal-year column.
#' @param duration_col Column name (character) of consecutive-day
#'   placement durations. Default `"NumberConsecutiveDays_Segregation"`.
#' @param year_col Column name of the fiscal-year identifier. Default
#'   `"EndFiscalYear"`.
#' @param id_col Column name of the per-year individual identifier.
#'   Default `"UniqueIndividual_ID"`. Required when `denominator` is
#'   anything other than `"row"`.
#' @param threshold_days Mandela duration threshold in days. Default
#'   `15` (UN Standard Minimum Rules).
#' @param denominator One of `"row"` (per-placement), `"individual_any"`
#'   (proportion of individuals with any placement above threshold),
#'   or `"individual_cumulative"` (proportion of individuals whose
#'   total within-year segregation days exceed threshold). Default
#'   `"individual_any"`.
#' @param broader_rc Logical. If `TRUE`, the numerator additionally
#'   counts placements with alert-complexity `>= 2` (using the three
#'   alert columns named in `alert_cols`) regardless of meeting
#'   `threshold_days`. Default `FALSE`.
#' @param alert_cols Character vector of binary alert columns used to
#'   compute alert-complexity for the broader rate. Default the three
#'   b01 alert columns.
#' @param meaningful_contact_col Optional. Column name of a
#'   1-if-meaningful-contact indicator (federal Sprott-Doob style).
#'   When supplied, rows with met-contact are excluded from the
#'   numerator.
#'
#' @return A data.frame with columns:
#'   \describe{
#'     \item{year}{Fiscal year (or `"pooled"`).}
#'     \item{denominator}{Total denominator under the chosen convention.}
#'     \item{n_mandela}{Numerator: count of records (or individuals)
#'       classified as Mandela-prolonged.}
#'     \item{rate}{Proportion `n_mandela / denominator` in $[0, 1]$.}
#'     \item{pct}{Same as `rate` expressed as percentage.}
#'     \item{n_broader_rc}{Broader-rate numerator (if `broader_rc`).}
#'     \item{rate_broader}{Broader-rate proportion (if `broader_rc`).}
#'   }
#'
#' @references
#' United Nations General Assembly (2015). United Nations Standard
#' Minimum Rules for the Treatment of Prisoners (the Nelson Mandela
#' Rules). A/RES/70/175.
#'
#' Sprott, J. B., & Doob, A. N. (2023). The (Non-)Use of Structured
#' Intervention Units in Canada. Canadian Journal of Criminology and
#' Criminal Justice, 65(1).
#'
#' @export
#' @examples
#' # Strict provincial Mandela on b01:
#' #   mrm_classify_mandela(b01_data)
#' #
#' # Broader restrictive-confinement (adds high-alert placements):
#' #   mrm_classify_mandela(b01_data, broader_rc = TRUE)
mrm_classify_mandela <- function(
  data,
  duration_col = "NumberConsecutiveDays_Segregation",
  year_col = "EndFiscalYear",
  id_col = "UniqueIndividual_ID",
  threshold_days = 15L,
  denominator = c("individual_any", "row", "individual_cumulative"),
  broader_rc = FALSE,
  alert_cols = c("MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert"),
  meaningful_contact_col = NULL
) {
  denominator <- match.arg(denominator)
  stopifnot(is.data.frame(data))
  stopifnot(duration_col %in% names(data))
  stopifnot(year_col %in% names(data))
  if (denominator != "row") {
    stopifnot(id_col %in% names(data))
  }

  # Strict per-row Mandela indicator
  dur <- data[[duration_col]]
  strict_row <- !is.na(dur) & dur > threshold_days

  # Broader: also count high-alert (ac >= 2) placements
  if (broader_rc) {
    stopifnot(all(alert_cols %in% names(data)))
    alerts_count <- rowSums(
      sapply(alert_cols, function(c) as.integer(data[[c]] > 0))
    )
    broader_row <- strict_row | (alerts_count >= 2L & !is.na(dur) &
                                  dur > threshold_days)
  } else {
    broader_row <- strict_row
  }

  # Meaningful-contact federal-style exclusion
  if (!is.null(meaningful_contact_col)) {
    stopifnot(meaningful_contact_col %in% names(data))
    met <- as.integer(data[[meaningful_contact_col]]) == 1L
    strict_row <- strict_row & !met
    broader_row <- broader_row & !met
  }

  years <- sort(unique(data[[year_col]]))
  rows <- lapply(c(as.list(years), list("pooled")), function(y) {
    if (identical(y, "pooled")) {
      mask <- rep(TRUE, nrow(data))
      label <- "pooled"
    } else {
      mask <- data[[year_col]] == y
      label <- y
    }
    sub <- data[mask, , drop = FALSE]
    sub_strict <- strict_row[mask]
    sub_broader <- broader_row[mask]

    if (denominator == "row") {
      denom <- sum(mask)
      n_m <- sum(sub_strict)
      n_b <- sum(sub_broader)
    } else if (denominator == "individual_any") {
      ids <- unique(sub[[id_col]])
      ids_m <- unique(sub[[id_col]][sub_strict])
      ids_b <- unique(sub[[id_col]][sub_broader])
      denom <- length(ids)
      n_m <- length(ids_m)
      n_b <- length(ids_b)
    } else {  # individual_cumulative
      ids <- unique(sub[[id_col]])
      cum_dur <- tapply(dur[mask], sub[[id_col]], sum, na.rm = TRUE)
      denom <- length(ids)
      n_m <- sum(cum_dur > threshold_days)
      # broader cumulative: same threshold, since alerts don't change duration
      n_b <- n_m
    }
    data.frame(
      year = as.character(label),
      denominator = denom,
      n_mandela = n_m,
      rate = if (denom > 0) n_m / denom else NA_real_,
      pct = if (denom > 0) round(100 * n_m / denom, 2) else NA_real_,
      n_broader_rc = n_b,
      rate_broader = if (denom > 0) n_b / denom else NA_real_
    )
  })
  do.call(rbind, rows)
}
