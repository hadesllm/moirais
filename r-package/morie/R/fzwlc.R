# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: Smoothed Wilcoxon signed-rank test (Ch 5)
#'
#' \eqn{W_n=\sum_i \mathrm{sign}(D_i) R_{\mathrm{smooth}}(|D_i|)},
#' z = W_n / sqrt(n(n+1)(2n+1)/6) ~ N(0,1).
#'
#' @param x Numeric vector.
#' @param theta0 Null centre; default 0.
#' @param h Bandwidth; default = Silverman.
#' @param alternative "two-sided", "greater", "less".
#' @return Named list with statistic, z, p_value, theta0, h, n, method.
#' @importFrom stats pnorm
#' @export
fzwlc <- function(x, theta0 = 0, h = NULL, alternative = "two-sided") {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                           method = "fzwlc - too few obs"))
  d <- x - theta0; ad <- abs(d); nz <- ad > 1e-12
  d <- d[nz]; ad <- ad[nz]; n_eff <- length(d)
  if (n_eff < 5L) return(list(statistic = NA_real_, p_value = NA_real_,
                                n = n_eff,
                                method = "fzwlc - too few nonzero"))
  if (is.null(h)) h <- .morie_silverman_h(ad)
  D <- outer(ad, ad, function(a, b) (a - b) / h)
  R_smooth <- rowSums(stats::pnorm(D))
  W_n <- sum(sign(d) * R_smooth)
  var <- n_eff * (n_eff + 1) * (2 * n_eff + 1) / 6
  z <- W_n / sqrt(var)
  p <- switch(alternative,
              "two-sided" = 2 * (1 - stats::pnorm(abs(z))),
              "greater"   = 1 - stats::pnorm(z),
              "less"      = stats::pnorm(z),
              stop("alternative must be two-sided/greater/less"))
  list(statistic = W_n, z = z, p_value = p,
       theta0 = theta0, h = h, n = n_eff,
       method = sprintf("Fauzi smoothed Wilcoxon signed-rank (%s) (Ch 5)",
                        alternative))
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(200); r <- fzwlc(x, 0); stopifnot(r$p_value > 0.05)

#' @rdname fzwlc
#' @keywords internal
#' @export
fauzi_smoothed_wilcoxon <- fzwlc
