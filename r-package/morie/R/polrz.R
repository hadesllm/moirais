# SPDX-License-Identifier: MIT OR Apache-2.0

#' Polarization index (Cohen-d between two groups; Armstrong Ch 8)
#'
#' Two-group polarization P = |mean_R - mean_D| / pooled_sd. When both
#' groups are constant, falls back to overall SD.
#'
#' @param x Numeric vector of ideal points.
#' @param group Optional two-level grouping vector. Default: split at
#'   the sample median.
#' @return Named list with `estimate`, `mean_R`, `mean_D`, `sd_R`,
#'   `sd_D`, `pooled_sd`, `n_R`, `n_D`, `method`.
#' @export
polrz <- function(x, group = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, n = n,
                          method = "polarization_index"))
  if (is.null(group)) {
    med <- stats::median(x); g <- as.integer(x >= med)
  } else {
    if (length(group) != n) stop("group must have length(x)")
    uniq <- unique(group)
    if (length(uniq) != 2L) stop("group must have exactly 2 levels")
    g <- as.integer(group == uniq[2])
  }
  xR <- x[g == 1L]; xD <- x[g == 0L]
  if (length(xR) < 1L || length(xD) < 1L)
    return(list(estimate = NA_real_, n = n, method = "polarization_index"))
  mR <- mean(xR); mD <- mean(xD)
  sR <- if (length(xR) > 1L) stats::sd(xR) else 0
  sD <- if (length(xD) > 1L) stats::sd(xD) else 0
  nR <- length(xR); nD <- length(xD)
  pooled <- sqrt(((nR - 1) * sR^2 + (nD - 1) * sD^2)
                 / max(nR + nD - 2L, 1L))
  if (pooled <= 0) pooled <- if (n > 1L) stats::sd(x) else 0
  pol <- if (pooled > 0) abs(mR - mD) / pooled else NA_real_
  list(estimate = pol, mean_R = mR, mean_D = mD, sd_R = sR, sd_D = sD,
       pooled_sd = pooled, n_R = nR, n_D = nD,
       method = "polarization_index")
}

#' @keywords internal
#' @rdname polrz
#' @export
polarization_index <- polrz
