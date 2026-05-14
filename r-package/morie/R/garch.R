# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fit a GARCH(1,1) model to a return series
#'
#' \deqn{\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2.}
#'
#' @param x Numeric return series.
#' @return Named list with \code{omega, alpha, beta, persistence, loglik,
#'   conditional_variance, n, method}.
#' @export
garch_fit <- function(x) {
  r <- as.numeric(x) - mean(as.numeric(x))
  n <- length(r); if (n < 10) stop("Need >=10 obs.")
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
    fit <- rugarch::ugarchfit(spec, r, solver = "hybrid")
    p <- rugarch::coef(fit)
    return(list(omega = unname(p["omega"]), alpha = unname(p["alpha1"]),
                beta = unname(p["beta1"]),
                persistence = unname(p["alpha1"] + p["beta1"]),
                loglik = as.numeric(rugarch::likelihood(fit)),
                conditional_variance = as.numeric(rugarch::sigma(fit))^2,
                n = n,
                method = "GARCH(1,1) via rugarch"))
  }
  neg_ll <- function(p) {
    omega <- p[1]; alpha <- p[2]; beta <- p[3]
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1) return(1e10)
    s2 <- numeric(n); s2[1] <- var(r)
    for (t in 2:n) s2[t] <- max(omega + alpha * r[t - 1]^2 + beta * s2[t - 1], 1e-12)
    0.5 * sum(log(2 * pi * s2) + r^2 / s2)
  }
  var_r <- var(r)
  opt <- nlminb(c(var_r * 0.05, 0.1, 0.85), neg_ll,
                lower = c(1e-8, 1e-8, 1e-8),
                upper = c(var_r * 10, 0.999, 0.999))
  omega <- opt$par[1]; alpha <- opt$par[2]; beta <- opt$par[3]
  s2 <- numeric(n); s2[1] <- var_r
  for (t in 2:n) s2[t] <- omega + alpha * r[t - 1]^2 + beta * s2[t - 1]
  list(omega = omega, alpha = alpha, beta = beta,
       persistence = alpha + beta, loglik = -opt$objective,
       conditional_variance = s2, n = n,
       method = "GARCH(1,1) Gaussian MLE (base R)")
}
