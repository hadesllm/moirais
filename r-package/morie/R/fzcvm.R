# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Cramer-von Mises test with kernel-smoothed CDF (Ch 5)
#'
#' \eqn{W_n^2 = n \int (\hat F_h(t)-F_0(t))^2 dF_0(t)}.
#'
#' @param x Numeric vector.
#' @param cdf "norm" (default) or a function F0(t).
#' @param args List of dist args; default = MLE for "norm".
#' @param h Bandwidth; default = Silverman.
#' @return Named list: statistic, p_value, h, n, method.
#' @importFrom stats sd pnorm qnorm
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
fzcvm <- function(x, cdf = "norm", args = NULL, h = NULL) {
  x <- as.numeric(x); n <- length(x)
  if (n < 5L) return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                           method = "fzcvm - too few obs"))
  if (is.null(h)) h <- .morie_silverman_h(x)
  if (is.function(cdf)) {
    t_grid <- seq(min(x), max(x), length.out = max(200L, n))
    F_ref  <- vapply(t_grid, cdf, numeric(1))
  } else if (identical(cdf, "norm")) {
    if (is.null(args)) args <- list(mean(x), stats::sd(x))
    u <- (seq_len(n) - 0.5) / n
    t_grid <- stats::qnorm(u, mean = args[[1]], sd = args[[2]])
    F_ref  <- stats::pnorm(t_grid, mean = args[[1]], sd = args[[2]])
  } else stop("supply a function for non-normal cdf")
  F_hat <- vapply(t_grid, function(g) mean(stats::pnorm((g - x) / h)),
                  numeric(1))
  w2 <- n * mean((F_hat - F_ref)^2)
  p <- .morie_cvm_pvalue(w2 / n)
  list(statistic = w2, p_value = p, h = h, n = n,
       method = "Fauzi kernel-smoothed Cramer-von Mises (Ch 5)")
}

.morie_cvm_pvalue <- function(w2) {
  if (w2 <= 0) return(1.0)
  tbl <- list(c(0.347, 0.10), c(0.461, 0.05), c(0.581, 0.025),
              c(0.743, 0.01), c(1.168, 0.001))
  if (w2 < tbl[[1]][1]) return(0.5)
  if (w2 > tbl[[length(tbl)]][1]) return(tbl[[length(tbl)]][2] * 0.5)
  for (i in seq_len(length(tbl) - 1)) {
    a <- tbl[[i]]; b <- tbl[[i + 1]]
    if (w2 >= a[1] && w2 <= b[1]) {
      lp <- log(a[2]) + (log(b[2]) - log(a[2])) * (w2 - a[1]) / (b[1] - a[1])
      return(exp(lp))
    }
  }
  0.5
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(500); r <- fzcvm(x, cdf = "norm", args = list(0,1))
# stopifnot(r$statistic >= 0)

#' @rdname fzcvm
#' @keywords internal
#' @export
fauzi_cvm_smoothed <- fzcvm
