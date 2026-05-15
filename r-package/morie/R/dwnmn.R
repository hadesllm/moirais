# SPDX-License-Identifier: AGPL-3.0-or-later

#' Dynamic ideal points / random-walk smoother (Armstrong Ch 6)
#'
#' Kalman + RTS smoother applied to a per-period scalar ideal-point
#' series (or a panel of legislators).
#'
#' @param x Numeric vector (per-period ideal points) or matrix
#'   (n_legislators by T).
#' @param sigma_w Random-walk innovation SD.
#' @return Named list with `smoothed`, `raw`, `sigma_w`, `n_periods`,
#'   `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
dwnmn <- function(x, sigma_w = 0.1) {
  if (is.matrix(x)) {
    n <- nrow(x); T <- ncol(x); out <- matrix(0, n, T)
    for (i in seq_len(n))
      out[i, ] <- dwnmn(x[i, ], sigma_w = sigma_w)$smoothed
    return(list(smoothed = out, raw = x, sigma_w = sigma_w,
                n_periods = T, n_units = n,
                method = "dynamic_wnominate"))
  }
  raw <- as.numeric(x); T <- length(raw)
  if (T == 0L)
    return(list(smoothed = numeric(0), raw = numeric(0),
                sigma_w = sigma_w, n_periods = 0L,
                method = "dynamic_wnominate"))
  s2_obs <- stats::var(raw) + 1e-6
  m <- numeric(T); P <- numeric(T); m[1] <- raw[1]; P[1] <- s2_obs
  for (t in 2:T) {
    mp <- m[t - 1]; Pp <- P[t - 1] + sigma_w^2
    K <- Pp / (Pp + s2_obs)
    m[t] <- mp + K * (raw[t] - mp)
    P[t] <- (1 - K) * Pp
  }
  ms <- m; Ps <- P
  if (T >= 2L) for (t in (T - 1L):1L) {
    Pp <- P[t] + sigma_w^2
    J <- P[t] / Pp
    ms[t] <- m[t] + J * (ms[t + 1] - m[t])
    Ps[t] <- P[t] + J^2 * (Ps[t + 1] - Pp)
  }
  list(smoothed = ms, raw = raw, P_smoothed = Ps, sigma_w = sigma_w,
       n_periods = T, method = "dynamic_wnominate")
}

#' @keywords internal
#' @rdname dwnmn
#' @export
dynamic_wnominate <- dwnmn
