# SPDX-License-Identifier: AGPL-3.0-or-later

#' Prophet-style additive decomposition (linear trend + Fourier seasonality)
#'
#' @param x Numeric univariate series.
#' @param period Seasonal period. Default 12.
#' @return Named list with \code{trend, seasonal, residual, slope,
#'   intercept, fourier_terms, period, n, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
prophet_components <- function(x, period = 12) {
  y <- as.numeric(x); n <- length(y)
  if (n < max(2 * period, 6)) stop("Series too short.")
  t <- seq(0, n - 1)
  fit <- lm(y ~ t)
  intercept <- coef(fit)[1]; slope <- coef(fit)[2]
  trend <- fitted(fit)
  detr <- y - trend
  K <- 5
  Fmat <- do.call(cbind, lapply(seq_len(K), function(k)
    cbind(sin(2 * pi * k * t / period), cos(2 * pi * k * t / period))))
  fcoef <- lsfit(Fmat, detr, intercept = FALSE)$coef
  seasonal <- as.numeric(Fmat %*% fcoef)
  residual <- detr - seasonal
  list(trend = trend, seasonal = seasonal, residual = residual,
       slope = as.numeric(slope), intercept = as.numeric(intercept),
       fourier_terms = fcoef, period = period, n = n,
       method = "Prophet-style linear-trend + Fourier(K=5) seasonality")
}
