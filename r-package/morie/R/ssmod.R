# SPDX-License-Identifier: AGPL-3.0-or-later

#' Local-level state-space model (Kalman filter+smoother)
#'
#' @param x Numeric univariate series.
#' @return Named list with \code{filtered_state, filtered_state_variance,
#'   smoothed_state, loglik, Q, R, n, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
state_space_model <- function(x) {
  y <- as.numeric(x); n <- length(y)
  if (n < 4) stop("Need >=4 obs.")
  if (requireNamespace("dlm", quietly = TRUE)) {
    build <- function(p) dlm::dlmModPoly(order = 1,
                                         dV = exp(p[1]), dW = exp(p[2]))
    fit <- dlm::dlmMLE(y, parm = c(log(var(diff(y)) / 2),
                                   log(var(diff(y)) / 2)),
                       build = build)
    mod <- build(fit$par)
    f <- dlm::dlmFilter(y, mod)
    s <- dlm::dlmSmooth(f)
    return(list(filtered_state = as.numeric(f$m)[-1],
                filtered_state_variance = sapply(dlm::dlmSvd2var(f$U.C, f$D.C),
                                                  function(x) x[1, 1])[-1],
                smoothed_state = as.numeric(s$s)[-1],
                loglik = -fit$value,
                Q = exp(fit$par[2]),
                R = exp(fit$par[1]),
                n = n,
                method = "Local-level Kalman via dlm"))
  }
  Q <- var(diff(y)) / 2; R <- var(diff(y)) / 2
  a <- numeric(n); P <- numeric(n)
  a[1] <- y[1]; P[1] <- 1e7
  ll <- 0
  for (t in 2:n) {
    Pp <- P[t - 1] + Q
    v <- y[t] - a[t - 1]
    Fv <- Pp + R; K <- Pp / Fv
    a[t] <- a[t - 1] + K * v
    P[t] <- Pp - K * Pp
    ll <- ll + -0.5 * (log(2 * pi * Fv) + v^2 / Fv)
  }
  a_s <- a; P_s <- P
  for (t in (n - 1):1) {
    Pp <- P[t] + Q; J <- P[t] / Pp
    a_s[t] <- a[t] + J * (a_s[t + 1] - a[t])
    P_s[t] <- P[t] + J^2 * (P_s[t + 1] - Pp)
  }
  list(filtered_state = a, filtered_state_variance = P,
       smoothed_state = a_s, loglik = ll, Q = Q, R = R, n = n,
       method = "Local-level Kalman filter+smoother (base R)")
}
