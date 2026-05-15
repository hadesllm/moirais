# SPDX-License-Identifier: AGPL-3.0-or-later

#' Unobserved-components decomposition (trend + seasonal + irregular)
#'
#' @param x Numeric univariate series.
#' @param period Seasonal period (pass 0 to omit). Default 12.
#' @param trend Trend specification, "local level" or "local linear".
#' @return Named list with \code{trend, seasonal, irregular, loglik, n,
#'   period, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
unobserved_components <- function(x, period = 12, trend = "local linear") {
  y <- as.numeric(x); n <- length(y)
  if (n < max(2 * period, 6)) stop("Series too short.")
  if (period > 1) {
    dec <- stats::decompose(stats::ts(y, frequency = period),
                            type = "additive")
    mu <- as.numeric(dec$trend)
    mu[is.na(mu)] <- mean(mu, na.rm = TRUE)
    season <- as.numeric(dec$seasonal)
    irr <- y - mu - season
  } else {
    mu <- stats::filter(y, rep(1 / 5, 5), sides = 2)
    mu <- as.numeric(mu); mu[is.na(mu)] <- mean(mu, na.rm = TRUE)
    season <- numeric(n); irr <- y - mu
  }
  list(trend = mu, seasonal = season, irregular = irr,
       loglik = NA_real_, n = n, period = period,
       method = "Additive trend+seasonal decomposition (base R)")
}
