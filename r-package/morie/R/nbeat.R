# SPDX-License-Identifier: AGPL-3.0-or-later

#' N-BEATS-style polynomial + Fourier basis-expansion forecasting
#'
#' @param x Numeric history.
#' @param horizon Forecast horizon. Default 1.
#' @param n_trend Polynomial-trend degree. Default 3.
#' @param n_season Number of Fourier harmonics. Default 5.
#' @param period Seasonal period. Default 12.
#' @return Named list with \code{forecast, fitted, trend, seasonal,
#'   theta_trend, theta_seasonal, r2, n, horizon, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
nbeats_basis <- function(x, horizon = 1, n_trend = 3, n_season = 5,
                          period = 12) {
  y <- as.numeric(x); n <- length(y)
  if (n < n_trend + 2 * n_season + 2)
    stop("Series too short for chosen basis.")
  t <- seq(0, n - 1)
  Tmat <- sapply(0:n_trend, function(k) t^k)
  Smat <- do.call(cbind, lapply(seq_len(n_season), function(j)
    cbind(sin(2 * pi * j * t / period),
          cos(2 * pi * j * t / period))))
  Xmat <- cbind(Tmat, Smat)
  coef <- lsfit(Xmat, y, intercept = FALSE)$coef
  fitted_y <- as.numeric(Xmat %*% coef)
  tf <- seq(n, n + horizon - 1)
  Tf <- sapply(0:n_trend, function(k) tf^k)
  Sf <- do.call(cbind, lapply(seq_len(n_season), function(j)
    cbind(sin(2 * pi * j * tf / period),
          cos(2 * pi * j * tf / period))))
  Xf <- cbind(Tf, Sf)
  forecast <- as.numeric(Xf %*% coef)
  theta_trend <- coef[seq_len(n_trend + 1)]
  theta_season <- coef[(n_trend + 2):length(coef)]
  trend <- as.numeric(Tmat %*% theta_trend)
  seasonal <- as.numeric(Smat %*% theta_season)
  ss_tot <- sum((y - mean(y))^2)
  r2 <- if (ss_tot > 0) 1 - sum((y - fitted_y)^2) / ss_tot else NA_real_
  list(forecast = forecast, fitted = fitted_y,
       trend = trend, seasonal = seasonal,
       theta_trend = theta_trend, theta_seasonal = theta_season,
       r2 = r2, n = n, horizon = horizon,
       method = sprintf("N-BEATS basis: poly(P=%d) + Fourier(H=%d, period=%d)",
                         n_trend, n_season, period))
}
