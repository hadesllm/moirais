# SPDX-License-Identifier: GPL-2.0-only
#' Generalised Extreme Value fit by ML (Coles 2001)
#'
#' Fits F(x) = exp(-(1 + xi (x - mu)/sigma)^(-1/xi)) by maximum
#' likelihood, with numerical observed-information SEs.  Uses the
#' \code{evd} package when available; otherwise pure-R optim.
#'
#' @param x numeric vector of block maxima.
#' @return list: mu, sigma, xi, se_mu, se_sigma, se_xi, loglik, n, method.
#' @keywords internal
extvm <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L)
    return(list(estimate = NA_real_, n = n, method = "GEV (n<5)"))
  # GEV density (Coles xi convention)
  log_gev <- function(par, x) {
    mu <- par[1]; sigma <- exp(par[2]); xi <- par[3]
    z <- (x - mu) / sigma
    if (abs(xi) < 1e-8) {
      ll <- -log(sigma) - z - exp(-z)
    } else {
      arg <- 1 + xi * z
      if (any(arg <= 0)) return(rep(-1e10, length(x)))
      ll <- -log(sigma) - (1 + 1/xi) * log(arg) - arg^(-1/xi)
    }
    ll
  }
  nll <- function(par) -sum(log_gev(par, x))
  init <- c(mean(x), log(stats::sd(x)), 0.1)
  fit <- stats::optim(init, nll, method = "BFGS", hessian = TRUE)
  mu <- fit$par[1]; sigma <- exp(fit$par[2]); xi <- fit$par[3]
  loglik <- -fit$value
  # Hessian is wrt (mu, log_sigma, xi); convert to (mu, sigma, xi)
  J <- diag(c(1, sigma, 1))
  cov_mat <- tryCatch(J %*% solve(fit$hessian) %*% t(J),
                      error = function(e) matrix(NA, 3, 3))
  ses <- sqrt(pmax(diag(cov_mat), 0))
  list(mu = as.numeric(mu), sigma = as.numeric(sigma), xi = as.numeric(xi),
       se_mu = as.numeric(ses[1]), se_sigma = as.numeric(ses[2]),
       se_xi = as.numeric(ses[3]),
       loglik = as.numeric(loglik),
       estimate = as.numeric(mu), se = as.numeric(ses[1]),
       n = as.integer(n),
       method = "GEV MLE (Coles 2001)")
}

# CANONICAL TEST
# set.seed(0); x <- evd::rgumbel(500, loc = 10, scale = 2)  # or evd::rgev
# r <- extvm(x)
# stopifnot(abs(r$xi) < 0.2)  # Gumbel = GEV(xi=0)

#' @rdname extvm
#' @keywords internal
#' @export
extreme_value_gev <- extvm
