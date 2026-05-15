# SPDX-License-Identifier: AGPL-3.0-or-later

#' Internal helpers shared across the time-series-advanced suite.
#' Not exported; consumed by garch/midas/etc callables only.
#' @keywords internal
#' @name time_series_advanced_helpers
#' @importFrom stats var sd lm coef residuals fitted lsfit fft
#'   acf arima ar nlminb pnorm dnorm cor decompose ts filter quantile
NULL

# Null-coalescing helper used internally by Johansen fallback critical
# values; not exported.
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

.morie_beta_weights <- function(t1, t2, K) {
  k <- seq_len(K) / (K + 1)
  w <- (k^(t1 - 1)) * ((1 - k)^(t2 - 1))
  if (sum(w) > 0) w / sum(w) else rep(1 / K, K)
}
