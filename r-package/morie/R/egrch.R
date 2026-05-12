# SPDX-License-Identifier: GPL-2.0-only

#' EGARCH(1,1) asymmetric volatility model
#'
#' @inheritParams garch_fit
#' @return Named list with \code{omega, alpha, gamma, beta, loglik,
#'   conditional_variance, n, method}.
#' @export
egarch_model <- function(x) {
  r <- as.numeric(x) - mean(as.numeric(x)); n <- length(r)
  if (n < 20) stop("Need >=20 obs.")
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(
      variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
    fit <- rugarch::ugarchfit(spec, r, solver = "hybrid")
    p <- rugarch::coef(fit)
    return(list(omega = unname(p["omega"]),
                alpha = unname(p["alpha1"]),
                gamma = unname(p["gamma1"]),
                beta  = unname(p["beta1"]),
                loglik = as.numeric(rugarch::likelihood(fit)),
                conditional_variance = as.numeric(rugarch::sigma(fit))^2,
                n = n,
                method = "EGARCH(1,1) via rugarch"))
  }
  EZ <- sqrt(2 / pi)
  neg_ll <- function(p) {
    omega <- p[1]; alpha <- p[2]; gamma <- p[3]; beta <- p[4]
    if (abs(beta) >= 1) return(1e10)
    log_s2 <- numeric(n); log_s2[1] <- log(var(r) + 1e-12)
    for (t in 2:n) {
      z <- r[t - 1] / sqrt(exp(log_s2[t - 1]) + 1e-12)
      log_s2[t] <- omega + beta * log_s2[t - 1] + alpha * (abs(z) - EZ) + gamma * z
    }
    s2 <- exp(log_s2)
    0.5 * sum(log(2 * pi * s2) + r^2 / s2)
  }
  opt <- nlminb(c(0, 0.1, 0, 0.9), neg_ll,
                lower = c(-5, -1, -1, -0.999),
                upper = c(5, 1, 1, 0.999))
  log_s2 <- numeric(n); log_s2[1] <- log(var(r) + 1e-12)
  for (t in 2:n) {
    z <- r[t - 1] / sqrt(exp(log_s2[t - 1]) + 1e-12)
    log_s2[t] <- opt$par[1] + opt$par[4] * log_s2[t - 1] +
                 opt$par[2] * (abs(z) - EZ) + opt$par[3] * z
  }
  list(omega = opt$par[1], alpha = opt$par[2],
       gamma = opt$par[3], beta = opt$par[4],
       loglik = -opt$objective,
       conditional_variance = exp(log_s2), n = n,
       method = "EGARCH(1,1) Gaussian MLE (base R)")
}
