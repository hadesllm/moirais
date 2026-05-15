# SPDX-License-Identifier: AGPL-3.0-or-later
#' Bootstrap confidence interval (percentile, BCa, studentized)
#'
#' R parity of \code{morie.fn.btsrp.bootstrap_ci}.  Three methods are
#' supported: percentile, BCa (Efron 1987 JASA), and studentized
#' (Hall 1988 nested resampling).
#'
#' @param x numeric vector.
#' @param statistic function: \code{function(x) -> scalar}.  Default \code{mean}.
#' @param B integer.  Bootstrap replicates (default 2000).
#' @param alpha numeric.  Two-sided level (default 0.05).
#' @param method "percentile", "bca", or "studentized".
#' @param seed integer.
#' @return Named list with estimate, se, ci_lower, ci_upper, alpha, B, n, method.
#' @references
#' Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
#' Efron, B. (1987). Better bootstrap confidence intervals. JASA, 82(397), 171-185.
#' @keywords internal
btsrp <- function(x, statistic = NULL, B = 2000L, alpha = 0.05,
                  method = c("percentile", "bca", "studentized"),
                  seed = 42L) {
  method <- match.arg(method)
  x <- as.numeric(x)
  n <- length(x)
  if (n < 2L) {
    return(list(estimate = NA_real_, se = NA_real_,
                ci_lower = NA_real_, ci_upper = NA_real_,
                n = n, method = method))
  }
  if (is.null(statistic)) statistic <- mean
  set.seed(seed)
  theta_hat <- statistic(x)
  boot <- replicate(B, statistic(sample(x, n, replace = TRUE)))
  se <- stats::sd(boot)
  if (method == "percentile") {
    ci <- stats::quantile(boot, probs = c(alpha / 2, 1 - alpha / 2),
                          names = FALSE)
    lo <- ci[1]; hi <- ci[2]
  } else if (method == "bca") {
    z0 <- stats::qnorm(mean(boot < theta_hat))
    jk <- vapply(seq_len(n), function(i) statistic(x[-i]), numeric(1))
    jk_bar <- mean(jk)
    num <- sum((jk_bar - jk)^3)
    den <- 6 * (sum((jk_bar - jk)^2))^1.5
    a <- if (den > 0) num / den else 0
    z_lo <- stats::qnorm(alpha / 2)
    z_hi <- stats::qnorm(1 - alpha / 2)
    a1 <- stats::pnorm(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 <- stats::pnorm(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    ci <- stats::quantile(boot, probs = c(a1, a2), names = FALSE)
    lo <- ci[1]; hi <- ci[2]
  } else {                     # studentized
    B2 <- max(50, B %/% 10)
    t_stars <- numeric(B)
    for (b in seq_len(B)) {
      idx <- sample.int(n, n, replace = TRUE)
      xb <- x[idx]; theta_b <- statistic(xb)
      inner <- replicate(B2,
                         statistic(sample(xb, n, replace = TRUE)))
      se_b <- stats::sd(inner)
      t_stars[b] <- if (se_b > 0) (theta_b - theta_hat) / se_b else 0
    }
    qs <- stats::quantile(t_stars, probs = c(alpha / 2, 1 - alpha / 2),
                          names = FALSE)
    lo <- theta_hat - qs[2] * se
    hi <- theta_hat - qs[1] * se
  }
  list(estimate = as.numeric(theta_hat), se = as.numeric(se),
       ci_lower = as.numeric(lo), ci_upper = as.numeric(hi),
       alpha = alpha, B = as.integer(B), n = as.integer(n),
       method = paste0("Bootstrap CI (", method, ")"))
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(100)
# r <- btsrp(x, B = 500, seed = 0)
# stopifnot(abs(r$estimate - mean(x)) < 1e-9)
# stopifnot(r$ci_lower < r$estimate, r$estimate < r$ci_upper)

#' @rdname btsrp
#' @keywords internal
#' @export
bootstrap_ci <- btsrp
