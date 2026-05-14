# SPDX-License-Identifier: MIT OR Apache-2.0

#' Huber-M location estimator with profiled scale
#'
#' theta_n = argmax_theta P_n m(.; theta, eta_n), eta_n = MAD/0.6745.
#'
#' @param x Numeric vector.
#' @param y Ignored (API parity).
#' @param k Huber tuning constant (default 1.345).
#' @param max_iter Newton iterations cap (default 100).
#' @param tol Convergence tolerance (default 1e-10).
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 5; Huber (1981).
#' @export
ksr10_kosorok_m_estimator <- function(x, y = NULL, k = 1.345,
                                       max_iter = 100, tol = 1e-10) {
  x <- as.numeric(x)
  n <- length(x)
  eta <- stats::mad(x, constant = 1) / 0.6745
  if (eta == 0) eta <- stats::sd(x)
  if (eta == 0) eta <- 1
  theta <- stats::median(x)
  for (it in seq_len(max_iter)) {
    u   <- (x - theta) / eta
    psi <- pmin(pmax(u, -k), k)
    denom <- mean(abs(u) <= k)
    if (denom == 0) break
    step <- eta * mean(psi) / denom
    if (abs(step) < tol) { theta <- theta + step; break }
    theta <- theta + step
  }
  u   <- (x - theta) / eta
  psi <- pmin(pmax(u, -k), k)
  A <- mean(abs(u) <= k) / eta
  B <- mean(psi^2)
  se <- sqrt(B / (A^2) / n)
  list(
    estimate = theta, se = se, n = n,
    method   = sprintf("Huber-M location (k=%.3f) with profiled MAD/0.6745 scale", k)
  )
}

# CANONICAL TEST
# set.seed(0); ksr10_kosorok_m_estimator(rnorm(200))

#' @rdname ksr10_kosorok_m_estimator
#' @keywords internal
#' @export
kosorok_m_estimator <- ksr10_kosorok_m_estimator
