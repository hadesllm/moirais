# SPDX-License-Identifier: GPL-2.0-only

#' ARCH(1)-in-mean model
#'
#' @inheritParams garch_fit
#' @return Named list with \code{mu, delta, omega, alpha, loglik,
#'   conditional_variance, n, method}.
#' @export
arch_in_mean <- function(x) {
  y <- as.numeric(x); n <- length(y)
  if (n < 20) stop("Need >=20 obs.")
  neg_ll <- function(p) {
    mu <- p[1]; delta <- p[2]; omega <- p[3]; alpha <- p[4]
    if (omega <= 0 || alpha < 0 || alpha >= 0.999) return(1e10)
    s2 <- numeric(n); s2[1] <- max(var(y), 1e-10)
    eps <- numeric(n); eps[1] <- y[1] - mu - delta * sqrt(s2[1])
    for (t in 2:n) {
      s2[t] <- max(omega + alpha * eps[t - 1]^2, 1e-12)
      eps[t] <- y[t] - mu - delta * sqrt(s2[t])
    }
    0.5 * sum(log(2 * pi * s2) + eps^2 / s2)
  }
  var_y <- var(y)
  opt <- nlminb(c(mean(y), 0, var_y * 0.5, 0.2), neg_ll,
                lower = c(-10, -10, 1e-8, 1e-8),
                upper = c(10, 10, var_y * 10, 0.999))
  mu <- opt$par[1]; delta <- opt$par[2]
  omega <- opt$par[3]; alpha <- opt$par[4]
  s2 <- numeric(n); s2[1] <- var_y
  eps <- numeric(n); eps[1] <- y[1] - mu - delta * sqrt(s2[1])
  for (t in 2:n) {
    s2[t] <- omega + alpha * eps[t - 1]^2
    eps[t] <- y[t] - mu - delta * sqrt(s2[t])
  }
  list(mu = mu, delta = delta, omega = omega, alpha = alpha,
       loglik = -opt$objective,
       conditional_variance = s2, n = n,
       method = "ARCH(1)-in-mean Gaussian MLE (base R)")
}
