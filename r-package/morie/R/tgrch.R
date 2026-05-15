# SPDX-License-Identifier: AGPL-3.0-or-later

#' GJR-GARCH(1,1) threshold GARCH
#'
#' @inheritParams garch_fit
#' @return Named list with \code{omega, alpha, gamma, beta, persistence,
#'   loglik, conditional_variance, n, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
tgarch_model <- function(x) {
  r <- as.numeric(x) - mean(as.numeric(x)); n <- length(r)
  if (n < 20) stop("Need >=20 obs.")
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(
      variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
    fit <- rugarch::ugarchfit(spec, r, solver = "hybrid")
    p <- rugarch::coef(fit)
    return(list(omega = unname(p["omega"]),
                alpha = unname(p["alpha1"]),
                gamma = unname(p["gamma1"]),
                beta  = unname(p["beta1"]),
                persistence = unname(p["alpha1"] + 0.5 * p["gamma1"] + p["beta1"]),
                loglik = as.numeric(rugarch::likelihood(fit)),
                conditional_variance = as.numeric(rugarch::sigma(fit))^2,
                n = n,
                method = "GJR-GARCH(1,1) via rugarch"))
  }
  neg_ll <- function(p) {
    omega <- p[1]; alpha <- p[2]; gamma <- p[3]; beta <- p[4]
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + 0.5 * gamma + beta >= 1)
      return(1e10)
    s2 <- numeric(n); s2[1] <- var(r) + 1e-10
    for (t in 2:n) {
      I <- if (r[t - 1] < 0) 1 else 0
      s2[t] <- max(omega + (alpha + gamma * I) * r[t - 1]^2 + beta * s2[t - 1],
                   1e-12)
    }
    0.5 * sum(log(2 * pi * s2) + r^2 / s2)
  }
  var_r <- var(r)
  opt <- nlminb(c(var_r * 0.05, 0.05, 0.05, 0.85), neg_ll,
                lower = c(1e-8, 1e-8, -0.5, 1e-8),
                upper = c(var_r * 10, 0.5, 0.999, 0.999))
  omega <- opt$par[1]; alpha <- opt$par[2]
  gamma <- opt$par[3]; beta <- opt$par[4]
  s2 <- numeric(n); s2[1] <- var_r
  for (t in 2:n) {
    I <- if (r[t - 1] < 0) 1 else 0
    s2[t] <- omega + (alpha + gamma * I) * r[t - 1]^2 + beta * s2[t - 1]
  }
  list(omega = omega, alpha = alpha, gamma = gamma, beta = beta,
       persistence = alpha + 0.5 * gamma + beta,
       loglik = -opt$objective,
       conditional_variance = s2, n = n,
       method = "GJR-GARCH(1,1) Gaussian MLE (base R)")
}
