# SPDX-License-Identifier: MIT OR Apache-2.0

#' Scalar-covariate Cox proportional-hazards partial-likelihood MLE
#'
#' Newton-Raphson on
#' l(beta) = sum over events of (X_i beta - log sum over j in R(t_i) of exp(X_j beta)).
#' Breslow tie handling.
#'
#' @param x Numeric covariate vector.
#' @param t Numeric vector of observed times.
#' @param event Integer/logical vector (1 = event, 0 = censored).
#' @param tol Convergence tolerance (default 1e-10).
#' @param max_iter Newton iterations cap (default 100).
#' @return Named list with estimate, se, n, method.
#' @references Kosorok (2008), Ch 8.
#' @export
ksr19_kosorok_cox_partial_likelihood <- function(x, t, event,
                                                  tol = 1e-10,
                                                  max_iter = 100) {
  x <- as.numeric(x); t <- as.numeric(t); event <- as.integer(event)
  n <- length(x)
  ord <- order(t)
  x <- x[ord]; t <- t[ord]; event <- event[ord]
  beta <- 0
  rev_cumsum <- function(v) rev(cumsum(rev(v)))
  for (it in seq_len(max_iter)) {
    wt <- exp(beta * x)
    S0 <- rev_cumsum(wt)
    S1 <- rev_cumsum(wt * x)
    S2 <- rev_cumsum(wt * x * x)
    grad <- sum(event * (x - S1 / S0))
    info <- sum(event * (S2 / S0 - (S1 / S0)^2))
    if (info <= 0) break
    step <- grad / info
    beta <- beta + step
    if (abs(step) < tol) break
  }
  wt <- exp(beta * x)
  S0 <- rev_cumsum(wt); S1 <- rev_cumsum(wt * x); S2 <- rev_cumsum(wt * x * x)
  info <- sum(event * (S2 / S0 - (S1 / S0)^2))
  se <- if (info > 0) sqrt(1 / info) else NA_real_
  list(
    estimate = beta,
    se       = se,
    n        = n,
    method   = "Cox PH partial-likelihood MLE (Breslow ties, scalar covariate)"
  )
}

# CANONICAL TEST
# set.seed(0); xs <- rnorm(100); ts <- rexp(100, rate=exp(0.5*xs))
# ksr19_kosorok_cox_partial_likelihood(xs, ts, rep(1, 100))

#' @rdname ksr19_kosorok_cox_partial_likelihood
#' @keywords internal
#' @export
kosorok_cox_partial_likelihood <- ksr19_kosorok_cox_partial_likelihood
