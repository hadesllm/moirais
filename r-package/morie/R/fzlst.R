# SPDX-License-Identifier: GPL-2.0-only

#' Fauzi: L-statistic for kernel functionals (Ch 5)
#'
#' \eqn{L_n = \sum_i c_{n,i} X_{(i)}} with
#' \eqn{c_{n,i} = \int_{(i-1)/n}^{i/n} J(u) du}.
#' Default J(u)=1 gives the sample mean.
#'
#' @param x Numeric vector.
#' @param score Function J(u); default constant 1.
#' @param n_quad Quadrature grid for variance.
#' @return Named list with estimate, se, n, method.
#' @importFrom stats sd quantile dnorm
#' @export
fzlst <- function(x, score = NULL, n_quad = 200L) {
  x <- as.numeric(x); n <- length(x)
  if (n < 2L) return(list(estimate = NA_real_, n = n,
                           method = "fzlst - too few obs"))
  if (is.null(score)) score <- function(u) rep(1, length(u))
  x_sorted <- sort(x)
  fine_n <- n * 16L
  fine_u <- seq(0, 1, length.out = fine_n + 1L)
  J_fine <- score(fine_u)
  edges <- seq(0, 1, length.out = n + 1L)
  cells <- findInterval(edges, fine_u, all.inside = TRUE)
  cells[length(cells)] <- length(fine_u) - 1L  # last edge
  weights <- numeric(n)
  for (i in seq_len(n)) {
    a <- cells[i]; b <- cells[i + 1L]
    if (b <= a) b <- a + 1L
    seg_u <- fine_u[a:(b + 1L)]
    seg_J <- J_fine[a:(b + 1L)]
    # trapezoidal rule
    weights[i] <- sum((seg_J[-length(seg_J)] + seg_J[-1L]) *
                        diff(seg_u)) / 2
  }
  L <- sum(weights * x_sorted)

  uu <- (seq_len(n_quad) - 0.5) / n_quad
  Q  <- as.numeric(stats::quantile(x, uu, names = FALSE))
  sigma <- stats::sd(x); if (sigma <= 0) sigma <- 1
  h <- 1.06 * sigma * n^(-1/5)
  f_Q <- vapply(Q, function(q) mean(stats::dnorm((q - x) / h) / h),
                numeric(1))
  J_at_u <- score(uu)
  U <- outer(uu, uu, pmin)
  Kmat <- (U - outer(uu, uu)) / (outer(f_Q, f_Q) + 1e-12)
  JJ <- outer(J_at_u, J_at_u)
  var <- mean(JJ * Kmat) / n
  var <- max(var, 0)
  list(estimate = L, se = sqrt(var), n = n,
       method = "Fauzi L-statistic with user score function J (Ch 5)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500); r <- fzlst(x)
# stopifnot(abs(r$estimate - mean(x)) < 1e-6)

#' @rdname fzlst
#' @keywords internal
#' @export
fauzi_l_statistic <- fzlst
