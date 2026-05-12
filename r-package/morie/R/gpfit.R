# SPDX-License-Identifier: GPL-2.0-only
#' Generalised Pareto fit (POT) by ML (Pickands 1975)
#'
#' Fits F(x) = 1 - (1 + xi x/sigma)^(-1/xi) to threshold exceedances.
#'
#' @param x numeric vector of raw observations.
#' @param threshold numeric; default 90th percentile.
#' @return list: scale (sigma), shape (xi), threshold, n_exceedances,
#'   se_sigma, se_xi, loglik, method.
#' @keywords internal
gpfit <- function(x, threshold = NULL) {
  x <- as.numeric(x)
  if (length(x) < 5L)
    return(list(estimate = NA_real_, n = length(x), method = "GP (n<5)"))
  if (is.null(threshold)) threshold <- stats::quantile(x, 0.90, names = FALSE)
  excess <- x[x > threshold] - threshold
  n <- length(excess)
  if (n < 5L)
    return(list(estimate = NA_real_, n = n,
                method = "GP (too few exceedances)"))
  log_gp <- function(par, y) {
    sigma <- exp(par[1]); xi <- par[2]
    if (abs(xi) < 1e-8) {
      ll <- -log(sigma) - y / sigma
    } else {
      arg <- 1 + xi * y / sigma
      if (any(arg <= 0)) return(rep(-1e10, length(y)))
      ll <- -log(sigma) - (1 + 1/xi) * log(arg)
    }
    ll
  }
  nll <- function(par) -sum(log_gp(par, excess))
  fit <- stats::optim(c(log(stats::sd(excess)), 0.1), nll,
                       method = "BFGS", hessian = TRUE)
  sigma <- exp(fit$par[1]); xi <- fit$par[2]
  loglik <- -fit$value
  J <- diag(c(sigma, 1))
  cov_mat <- tryCatch(J %*% solve(fit$hessian) %*% t(J),
                      error = function(e) matrix(NA, 2, 2))
  ses <- sqrt(pmax(diag(cov_mat), 0))
  list(scale = as.numeric(sigma), shape = as.numeric(xi),
       threshold = as.numeric(threshold), n_exceedances = as.integer(n),
       se_sigma = as.numeric(ses[1]), se_xi = as.numeric(ses[2]),
       loglik = as.numeric(loglik),
       estimate = as.numeric(sigma), se = as.numeric(ses[1]),
       method = "GP MLE (Pickands 1975)")
}

# CANONICAL TEST
# set.seed(0); x <- rexp(2000, rate = 1)
# r <- gpfit(x, threshold = 0.5)
# stopifnot(abs(r$shape) < 0.2)  # exponential = GP(xi=0)

#' @rdname gpfit
#' @keywords internal
#' @export
generalized_pareto <- gpfit
